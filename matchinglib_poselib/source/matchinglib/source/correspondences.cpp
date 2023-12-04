// Released under the MIT License - https://opensource.org/licenses/MIT
//
// Copyright (c) 2019 AIT Austrian Institute of Technology GmbH
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)

#include "matchinglib_correspondences.h"
#include "matchinglib_imagefeatures.h"
#include "matchinglib_matchers.h"
#include "match_statOptFlow.h"
#include "gms.h"
#include "alphanum.hpp"
#include "matchinglib/random_numbers.h"
#include <trees.h>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <random>
#include "opencv2/imgproc.hpp"
#include <opencv2/xfeatures2d.hpp>

#include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/nonfree/nonfree.hpp"
// #include <opencv2/nonfree/features2d.hpp>

#define USE_FSTREAM 0
#if USE_FSTREAM
#include <fstream>
#include <bitset>
#include <type_traits>
#endif

#ifdef WITH_AKAZE_CUDA
#include <AKAZE.h>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_affine_feature.h>
#endif

#define USE_CLAHE_FOR_HIST_EQU 1

#define DBG_SHOW_MATCHES 0
#define DBG_SHOW_KEYPOINTS 0

using namespace cv;
using namespace std;

namespace matchinglib
{

  /* --------------------- Function prototypes --------------------- */
#if USE_FSTREAM
  // SFINAE for safety. Sue me for putting it in a macro for brevity on the function
#define IS_INTEGRAL(T) typename std::enable_if<std::is_integral<T>::value>::type * = 0
  void writeDescriptorsToDisk(cv::Mat descriptors, std::string file);
  template <class T>
  std::string integral_to_binary_string(T byte, IS_INTEGRAL(T))
  {
    std::bitset<sizeof(T) * CHAR_BIT> bs(byte);
    return bs.to_string();
  }
#endif
#if DBG_SHOW_MATCHES
  void drawMatchesImgPair(const cv::Mat &img1, const cv::Mat &img2, 
                          const std::vector<cv::KeyPoint> &kp1, const std::vector<cv::KeyPoint> &kp2, 
                          const std::vector<cv::DMatch> &matches, 
                          const int &img1_idx, 
                          const int &img2_idx, 
                          const size_t &nrMatchesLimit = 0);
#endif
  void visualizeKeypoints(const cv::Mat &img, 
                          const std::vector<cv::KeyPoint> &keypoints, 
                          const std::string &img_name, 
                          const std::string &feature_type = "");
  void visualizeKeypoints(const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors,
                          const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap, 
                          const std::string &feature_type = "");
  void visualizeKeypoints(const std::unordered_map<int, std::vector<cv::KeyPoint>> &keypoints_combined, 
                          const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap, 
                          const std::string &feature_type = "");
  void visualizeKeypoints(const std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &init_keypoints, 
                          const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap, 
                          const std::string &descriptor_type = "");
  void visualizeImg(const cv::Mat &img, const std::string &img_baseName, const int img_idx);
  cv::Ptr<cv::FeatureDetector> createDetector(const string &keypointtype, const int limitNrfeatures = 10000);
  cv::Ptr<cv::DescriptorExtractor> createExtractor(std::string const &descriptortype, std::string const &keypointtype, const int &nrFeaturesMax = 2000);
#ifdef WITH_AKAZE_CUDA
  cv::Ptr<cv::cuda::Feature2DAsync> createCudaDetector(const string &keypointtype, const int limitNrfeatures = 10000);
  cv::Ptr<cv::cuda::Feature2DAsync> createCudaExtractor(std::string const &descriptortype, const int &nrFeaturesMax = 2000);
  bool IsFeatureCudaTypeSupported(const std::string &type);
  std::vector<std::string> GetSupportedFeatureCudaTypes();
#endif
  void getKeypointsThreadFunc(const int startIdx, 
                              const int endIdx, 
                              const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                              const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                              cv::Ptr<cv::FeatureDetector> detector, 
                              std::shared_ptr<std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>>> keypoints, 
                              std::exception_ptr &thread_exception, 
                              const int limitNrFeatures = 8000);
#ifdef WITH_AKAZE_CUDA
  void getKeypointsCudaThreadFunc(const int startIdx, 
                                  const int endIdx, 
                                  const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                                  const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                  cv::Ptr<cv::cuda::Feature2DAsync> detector, 
                                  std::shared_ptr<std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>>> keypoints, 
                                  std::exception_ptr &thread_exception);
  void getKeypointsAkazeCudaThreadFunc(const int startIdx, 
                                       const int endIdx, 
                                       const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                                       const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                       cv::Ptr<matchinglib::cuda::Feature2DAsync> detector, 
                                       std::shared_ptr<std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>>> keypoints, 
                                       std::exception_ptr &thread_exception, 
                                       std::mt19937 mt);
#endif
  void getDescriptorsThreadFunc(const int startIdx, 
                                const int endIdx, 
                                const std::string descr_type, 
                                const std::vector<int> indices_threads, 
                                const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                cv::Ptr<cv::DescriptorExtractor> extractor, 
                                std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors, 
                                std::exception_ptr &thread_exception);
#ifdef WITH_AKAZE_CUDA
  void getDescriptorsCudaThreadFunc(const int startIdx, 
                                    const int endIdx, 
                                    const std::string descr_type, 
                                    const std::vector<int> indices_threads, 
                                    const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                    cv::Ptr<cv::cuda::Feature2DAsync> extractor, 
                                    std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                    std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors, 
                                    std::exception_ptr &thread_exception);
  void getDescriptorsAkazeCudaThreadFunc(const int startIdx, 
                                         const int endIdx, 
                                         const std::vector<int> indices_threads, 
                                         const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                         cv::Ptr<matchinglib::cuda::Feature2DAsync> extractor, 
                                         std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                         std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors, 
                                         std::exception_ptr &thread_exception, 
                                         std::mt19937 mt);
#endif
  void getFeaturesThreadFunc(const int startIdx, 
                             const int endIdx, 
                             const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                             const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                             cv::Ptr<cv::Feature2D> kp_detect_ptr, 
                             std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                             std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors);
#ifdef WITH_AKAZE_CUDA
  void getFeaturesCudaThreadFunc(const int startIdx, 
                                 const int endIdx, 
                                 const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                                 const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                 cv::Ptr<cv::cuda::Feature2DAsync> kp_detect_ptr, 
                                 std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                 std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors);
  void getFeaturesAkazeCudaThreadFunc(const int startIdx, 
                                      const int endIdx, 
                                      const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                                      const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                      cv::Ptr<matchinglib::cuda::Feature2DAsync> kp_detect_ptr, 
                                      std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                      std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors, 
                                      std::mt19937 mt);
#endif
  void getFeaturesAffineThreadFunc(const int startIdx, 
                                   const int endIdx, 
                                   const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                                   const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                   cv::Ptr<cv::Feature2D> kp_detect_ptr, 
                                   std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                   std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors);
  void filterResponseAffineThreadFunc(const int startIdx, 
                                      const int endIdx, 
                                      const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> *keypoints_descriptors, 
                                      std::shared_ptr<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>> filtered_features);
  void filterResponseAreaBasedAffineThreadFunc(const int startIdx, 
                                               const int endIdx, 
                                               const int limitNrFeatures, 
                                               const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> *keypoints_descriptors, 
                                               const cv::Size imgSize, 
                                               std::shared_ptr<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>> filtered_features);
  void filterResponseAreaBasedAffineThreadFunc(const int startIdx, 
                                               const int endIdx, 
                                               const std::unordered_map<int, int> &individual_limits, 
                                               const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> *keypoints_descriptors, 
                                               const cv::Size imgSize, 
                                               std::shared_ptr<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>> filtered_features);

  /* --------------------- Functions --------------------- */

  int getCorrespondences(Mat &img1,
                         Mat &img2,
                         std::vector<cv::DMatch> &finalMatches,
                         std::vector<cv::KeyPoint> &kp1,
                         std::vector<cv::KeyPoint> &kp2,
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
    std::random_device rd;
    std::mt19937 g(rd());
    return getCorrespondences(img1, img2, finalMatches, kp1, kp2, g, featuretype, extractortype, matchertype, dynamicKeypDet, limitNrfeatures, VFCrefine, GMSrefine, ratioTest, SOFrefine, subPixRefine, verbose, idxPars_NMSLIB, queryPars_NMSLIB);
  }

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
                         std::vector<cv::DMatch> &finalMatches,
                         std::vector<cv::KeyPoint> &kp1,
                         std::vector<cv::KeyPoint> &kp2,
                         std::mt19937 &mt,
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
    if (img1.empty() || img2.empty())
    {
      cout << "No image information provided!" << endl;
      return -1;
    }

    if (((img1.rows != img2.rows) || (img1.cols != img2.cols)) && !matchertype.compare("GMBSOF"))
    {
      cout << "Images should have the same size when using GMBSOF! There might be errors!" << endl;
      // return -1;
    }

    if (((img1.rows != img2.rows) || (img1.cols != img2.cols)) && SOFrefine)
    {
      cout << "Images should have the same size when using SOF-filtering! There might be errors!" << endl;
      // return -1;
    }

    if ((!featuretype.compare("MSER") || !featuretype.compare("SimpleBlob")) && !matchertype.compare("GMBSOF"))
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

    if (verbose > 1)
    {
      t_mea = (double)getTickCount(); // Start time measurement
    }

    if (getKeypoints(img1, keypoints1, featuretype, dynamicKeypDet, limitNrfeatures) != 0)
    {
      return -2; // Error while calculating keypoints
    }

    if (getKeypoints(img2, keypoints2, featuretype, dynamicKeypDet, limitNrfeatures) != 0)
    {
      return -2; // Error while calculating keypoints
    }

    if (verbose > 1)
    {
      t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); // End time measurement
      cout << "Time for keypoint detection (2 imgs): " << t_mea << "ms" << endl;
      t_oa = t_mea;
      t_mea = (double)getTickCount(); // Start time measurement
    }

    bool onlyKeypoints = false;

    if (matchertype == "LKOF" || matchertype == "LKOFT")
    {
      onlyKeypoints = true;

      if (matchertype == "LKOF")
      {
        matchinglib::getMatches_OpticalFlow(keypoints1, keypoints2,
                                            img1, img2,
                                            finalMatches,
                                            false, false,
                                            cv::Size(11, 11), 5.0f);
      }

      if (matchertype == "LKOFT")
      {
        matchinglib::getMatches_OpticalFlowTracker(keypoints1, keypoints2, cv::Mat(),
                                                   img1, img2, finalMatches,
                                                   "LKOFT", "ORB",
                                                   false, false,
                                                   cv::Size(11, 11));

        keypoints2 = keypoints1; // SAME ASSIGNEMNENT!
      }

      if (verbose > 0)
      {
        if (verbose > 1)
        {
          t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); // End time measurement
          cout << "getMatches_OpticalFlow: " << t_mea << "ms" << endl;
          t_oa += t_mea;
          cout << "getMatches_OpticalFlow + keypoint detection: " << t_oa << "ms" << endl;

          if (verbose > 2)
          {
            cout << "# of features (1st img): " << keypoints1.size() << endl;
            cout << "# of features (2nd img): " << keypoints2.size() << endl;
          }
        }

        t_mea = (double)getTickCount(); // Start time measurement
      }
    }
    else
    {

      if (getDescriptors(img1, keypoints1, extractortype, descriptors1, featuretype) != 0)
      {
        return -3; // Error while extracting descriptors
      }

      if (getDescriptors(img2, keypoints2, extractortype, descriptors2, featuretype) != 0)
      {
        return -3; // Error while extracting descriptors
      }

#if USE_FSTREAM
      // Function to write descriptor entries to disk -> for external tuning of the paramters of matcher VPTREE
      writeDescriptorsToDisk(descriptors1, "C:\\work\\THIRDPARTY\\NMSLIB-1.6\\trunk\\sample_data\\descriptors_" + extractortype + ".txt");
      return -1;
#endif

      if (verbose > 0)
      {
        if (verbose > 1)
        {
          t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); // End time measurement
          cout << "Time for descriptor extraction (2 imgs): " << t_mea << "ms" << endl;
          t_oa += t_mea;
          cout << "Time for feature detection (2 imgs): " << t_oa << "ms" << endl;

          if (verbose > 2)
          {
            cout << "# of features (1st img): " << keypoints1.size() << endl;
            cout << "# of features (2nd img): " << keypoints2.size() << endl;
          }
        }

        t_mea = (double)getTickCount(); // Start time measurement
      }
    }

    if (matchertype == "ALKOF")
    {
      matchinglib::getMatches_OpticalFlowAdvanced(keypoints1, keypoints2,
                                                  descriptors1, descriptors2,
                                                  img1, img2,
                                                  finalMatches,
                                                  "ALKOF",
                                                  false, false,
                                                  cv::Size(11, 11), 5.0f, 3);
    }
    else if (matchertype == "ALKOFT")
    {
      matchinglib::getMatches_OpticalFlowTracker(keypoints1, keypoints2, descriptors1,
                                                 img1, img2,
                                                 finalMatches,
                                                 "ALKOFT",
                                                 "ORB",
                                                 false, false,
                                                 cv::Size(11, 11));
    }
    else if (!onlyKeypoints)
    {
      err = getMatches(keypoints1, keypoints2, descriptors1, descriptors2, imgSi, finalMatches, mt, matchertype, VFCrefine, ratioTest, extractortype, idxPars_NMSLIB, queryPars_NMSLIB);
    }

    if (err != 0)
    {
      return (-4 + err); // Matching failed
    }

    if (verbose > 0)
    {
      t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); // End time measurement
      cout << "Time for matching: " << t_mea << "ms" << endl;

      if (verbose > 1)
      {
        t_oa += t_mea;
        cout << "Time for feature detection (2 imgs) and matching: " << t_oa << "ms" << endl;

        if (verbose > 2)
        {
          cout << "# of matches: " << finalMatches.size() << endl;
        }
      }
    }

    if (GMSrefine)
    {
      if (verbose > 1)
      {
        t_mea = (double)getTickCount(); // Start time measurement
      }

      std::vector<bool> inlierMask;
      int n_gms = filterMatchesGMS(keypoints1, imgSi, keypoints2, imgSi, finalMatches, inlierMask, false, false);

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
        t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); // End time measurement
        cout << "Time GMS refinement: " << t_mea << "ms" << endl;
        t_oa += t_mea;
        cout << "Overall time with GMS refinement: " << t_oa << "ms" << endl;

        if (verbose > 2)
        {
          cout << "# of matches after GMS refinement: " << finalMatches.size() << endl;
        }
      }
    }

    if (SOFrefine)
    {
      if (!matchertype.compare("GMBSOF"))
      {
        cout << "SOF-filtering makes no sence when using GMBSOF! Skipping the filtering step." << endl;
      }
      else
      {
        if (verbose > 1)
        {
          t_mea = (double)getTickCount(); // Start time measurement
        }

        filterMatchesSOF(keypoints1, keypoints2, imgSi, finalMatches);

        if (verbose > 1)
        {
          t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); // End time measurement
          cout << "Time SOF refinement: " << t_mea << "ms" << endl;
          t_oa += t_mea;
          cout << "Overall time with SOF refinement: " << t_oa << "ms" << endl;

          if (verbose > 2)
          {
            cout << "# of matches after SOF refinement: " << finalMatches.size() << endl;
          }
        }
      }
    }

    if (subPixRefine)
    {
      if (verbose > 1)
      {
        t_mea = (double)getTickCount(); // Start time measurement
      }

      std::vector<bool> inliers;
      vector<int> queryIdxs(finalMatches.size()), trainIdxs(finalMatches.size());
      vector<cv::KeyPoint> keypoints1_tmp, keypoints2_tmp;
      std::vector<cv::DMatch> finalMatches_tmp;

      for (size_t i = 0; i < finalMatches.size(); i++)
      {
        queryIdxs[i] = finalMatches[i].queryIdx;
        trainIdxs[i] = finalMatches[i].trainIdx;
      }

      for (size_t i = 0; i < finalMatches.size(); i++)
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
        for (size_t i = 0; i < finalMatches.size(); i++)
        {
          keypoints2[trainIdxs[i]] = keypoints2_tmp[i];
        }

        for (int i = (int)finalMatches.size() - 1; i >= 0; i--)
        {
          if (inliers[i])
          {
            finalMatches_tmp.push_back(finalMatches[i]);
          }
        }

        finalMatches = finalMatches_tmp;
      }

      if (verbose > 1)
      {
        t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); // End time measurement
        cout << "Time subpixel refinement: " << t_mea << "ms" << endl;
        t_oa += t_mea;
        cout << "Overall time: " << t_oa << "ms" << endl;

        if (verbose > 2)
        {
          cout << "# of matches after subpixel refinement: " << finalMatches.size() << endl;
        }
      }
    }

    if (verbose > 0)
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
                        std::vector<cv::DMatch> &matches)
  {
    vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
    vector<cv::KeyPoint> keypoints1_tmp, keypoints2_tmp;
    cv::Mat inliers;
    std::vector<cv::DMatch> finalMatches_tmp;

    for (size_t i = 0; i < matches.size(); i++)
    {
      queryIdxs[i] = matches[i].queryIdx;
      trainIdxs[i] = matches[i].trainIdx;
    }

    for (size_t i = 0; i < matches.size(); i++)
    {
      keypoints1_tmp.push_back(keypoints1.at(queryIdxs[i]));
      keypoints2_tmp.push_back(keypoints2.at(trainIdxs[i]));
    }

    /* ONLY FOR DEBUGGING START */
    // Mat drawImg;
    // drawMatches( *img1, keypoints1, *img2, keypoints2, matches, drawImg );
    ////imwrite("C:\\work\\bf_matches_cross-check.jpg", drawImg);
    // cv::namedWindow( "Source_1", CV_WINDOW_NORMAL );
    // cv::imshow( "Source_1", drawImg );
    // cv::waitKey(0);

    // vector<char> matchesMask( matches.size(), 0 );
    /* ONLY FOR DEBUGGING END */

    EMatFloat2 keyP1(keypoints1_tmp.size(), 2), keyP2(keypoints2_tmp.size(), 2);

    for (unsigned int i = 0; i < keypoints1_tmp.size(); i++)
    {
      keyP1(i, 0) = keypoints1_tmp[i].pt.x;
      keyP1(i, 1) = keypoints1_tmp[i].pt.y;
    }

    for (unsigned int i = 0; i < keypoints2_tmp.size(); i++)
    {
      keyP2(i, 0) = keypoints2_tmp[i].pt.x;
      keyP2(i, 1) = keypoints2_tmp[i].pt.y;
    }

    std::vector<std::vector<cv::Point3f>> gridSearchParams;
    float gridElemSize;

    if (getStatisticalMatchingPositions(keyP1, keyP2, imgSi, gridSearchParams, &gridElemSize, inliers) != 0)
    {
      // Calculation of flow statistic failed
      cout << "Filtering with SOF failed! Taking unfiltered matches." << endl;
    }
    else
    {

      /* ONLY FOR DEBUGGING START */
      // for( size_t i1 = 0; i1 < keypoints1_tmp.size(); i1++)
      //{
      //   if( inliers.at<bool>(i1,0) == true )
      //   {
      //     matchesMask[i1] = 1;
      //   }
      // }
      // drawMatches( *img1, keypoints1, *img2, keypoints2, matches, drawImg, Scalar::all(-1)/*CV_RGB(0, 255, 0)*/, CV_RGB(0, 0, 255), matchesMask);
      ////imwrite("C:\\work\\bf_matches_filtered_stat_flow.jpg", drawImg);
      // cv::imshow( "Source_1", drawImg );
      // cv::waitKey(0);
      /* ONLY FOR DEBUGGING END */

      for (size_t i1 = 0; i1 < matches.size(); i1++)
      {
        if (inliers.at<bool>((int)i1, 0))
        {
          finalMatches_tmp.push_back(matches[i1]);
        }
      }

      if (finalMatches_tmp.size() >= MIN_FINAL_MATCHES)
      {
        matches = finalMatches_tmp;
      }
    }
  }

#if USE_FSTREAM
  // Function to write descriptor entries to disk -> for external tuning of the paramters of matcher VPTREE
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

  bool getMatch3Corrs(const cv::Point2f &pt1, const cv::Point2f &pt2, const cv::Mat &F1, const cv::Mat &F2, const cv::Mat descr1, const cv::Mat descr2, const FeatureKDTree &ft, cv::Mat &descr3, cv::Point2f &pt3, const double &descr_dist_max, const float r_sqrd)
  {
    // Two matching feature positions in 2 cams (c1, c2)
    cv::Mat x1 = (cv::Mat_<double>(3, 1) << static_cast<double>(pt1.x), static_cast<double>(pt1.y), 1.0);
    cv::Mat x2 = (cv::Mat_<double>(3, 1) << static_cast<double>(pt2.x), static_cast<double>(pt2.y), 1.0);
    // Calculate epipolar line in target image using first match and F (c1 -> c3)
    cv::Mat l31 = F1 * x1;
    // Calculate epipolar line in target image using second match and F (c2 -> c3)
    cv::Mat l32 = F2 * x2;
    // Calculate point where both epipolar lines meet
    cv::Mat x3 = l31.cross(l32);
    x3 /= x3.at<double>(2);
    const cv::Point2f pt312(static_cast<float>(x3.at<double>(0)), static_cast<float>(x3.at<double>(1)));

    // Search near the estimated point from above and check the descriptor distance
    cv::KeyPoint kp3;
    double dd1;
    if (!ft.getBestKeypointDescriptorMatch(pt312, descr1, kp3, descr3, dd1, descr_dist_max, r_sqrd))
    {
      return false;
    }
    const double dd2 = getDescriptorDist(descr2, descr3);
    if (dd2 > descr_dist_max)
    {
      return false;
    }
    pt3 = kp3.pt;
    return true;
  }

  void AffineMatchesFilterData::find_emplace(const int &c)
  {
      const std::lock_guard lock(m);
      if (kp_descr_valid_indices.find(c) == kp_descr_valid_indices.end())
      {
          kp_descr_valid_indices.emplace(c, std::set<int>());
      }
  }

  void AffineMatchesFilterData::emplaceIdx(const int &c, const int &val)
  {
      const std::lock_guard lock(m);
      if (kp_descr_valid_indices.find(c) != kp_descr_valid_indices.end())
      {
          kp_descr_valid_indices.at(c).emplace(val);
      }
  }

  Matching::Matching(const std::vector<std::string> &img_file_names,
                     const std::vector<std::string> &keypoint_types, 
                     const std::string &descriptor_type,
                     const std::vector<std::string> &mask_file_names,
                     const std::vector<int> &img_indices,
                     const bool sort_file_names,
                     const double &img_scale, 
                     const bool equalizeImgs, 
                     const int &nrFeaturesToExtract_,
                     const int &cpuCnt_) : img_scaling(img_scale), 
                                           keypoint_types_(keypoint_types), 
                                           descriptor_type_(descriptor_type), 
                                           equalizeImgs_(equalizeImgs), 
                                           nr_keypoint_types(keypoint_types_.size()), 
                                           cpuCnt(cpuCnt_),
                                           largest_cam_idx(static_cast<int>(img_file_names.size())),
                                           haveMasks(!mask_file_names.empty()),
                                           nrFeaturesToExtract(nrFeaturesToExtract_)
  {
    if (cpuCnt <= 0)
    {
        cpuCnt = cpuCnt == 0 ? 1 : -1 * cpuCnt;
        cpuCnt = std::max(static_cast<int>(std::thread::hardware_concurrency()) / cpuCnt, 1);
    }

    if(img_file_names.empty())
    {
      throw runtime_error("No image file names provided.");
    }
    if(haveMasks && img_file_names.size() != mask_file_names.size())
    {
      throw runtime_error("Number of masks must be the same as images.");
    }

    indices = img_indices.empty() ? vector<int>(largest_cam_idx):img_indices;
    if(img_indices.empty())
    {
      std::iota(indices.begin(), indices.end(), 0);
      if(sort_file_names)
      {
        // sort indexes based on comparing values in img_file_names (lambda function)
        stable_sort(indices.begin(), indices.end(), [&img_file_names](const int &i1, const int &i2) {return doj::alphanum_comp(img_file_names[i1], img_file_names[i2]) < 0;});
      }
      for(const auto &i : indices)
      {
        if(haveMasks)
        {
          img_mask_names.emplace_back(make_pair(img_file_names.at(i), mask_file_names.at(i)));
        }
        else
        {
          img_mask_names.emplace_back(make_pair(img_file_names.at(i), ""));
        }
      }
    }
    else
    {
      if(img_file_names.size() != indices.size())
      {
        throw runtime_error("Number of indices must be the same as images.");
      }
      std::vector<pair<int, int>> idx_tmp(largest_cam_idx);
      for (int i = 0; i < largest_cam_idx; i++)
      {
        idx_tmp.emplace_back(make_pair(i, indices[i]));
      }
      stable_sort(idx_tmp.begin(), idx_tmp.end(), [](const pair<int, int> &i1, const pair<int, int> &i2) {return i1.second < i2.second;});
      indices.clear();
      for(const auto &i: idx_tmp)
      {
        indices.emplace_back(i.second);
        if(haveMasks)
        {
          img_mask_names.emplace_back(make_pair(img_file_names.at(i.first), mask_file_names.at(i.first)));
        }
        else
        {
          img_mask_names.emplace_back(make_pair(img_file_names.at(i.first), ""));
        }
      }
    }
    for(const auto &i: indices)
    {
      imageMap.emplace(i, make_pair(cv::Mat(), cv::Mat()));
    }
    
    std::cout << "Loading image data ..." << endl;
    loadImages();
  }

  Matching::Matching(const std::vector<cv::Mat> &imgs,
                     const std::vector<std::string> &keypoint_types, 
                     const std::string &descriptor_type,
                     const std::vector<cv::Mat> &masks,
                     const std::vector<int> &img_indices,
                     const double &img_scale, 
                     const bool equalizeImgs, 
                     const int &nrFeaturesToExtract_,
                     const int &cpuCnt_) : img_scaling(img_scale), 
                                           keypoint_types_(keypoint_types), 
                                           descriptor_type_(descriptor_type), 
                                           equalizeImgs_(equalizeImgs), 
                                           nr_keypoint_types(keypoint_types_.size()), 
                                           cpuCnt(cpuCnt_),
                                           largest_cam_idx(static_cast<int>(imgs.size())),
                                           haveMasks(!masks.empty()),
                                           nrFeaturesToExtract(nrFeaturesToExtract_)
  {
    if (cpuCnt <= 0)
    {
        cpuCnt = cpuCnt == 0 ? 1 : -1 * cpuCnt;
        cpuCnt = std::max(static_cast<int>(std::thread::hardware_concurrency()) / cpuCnt, 1);
    }

    if(imgs.empty())
    {
      throw runtime_error("No images provided.");
    }
    if(haveMasks && imgs.size() != masks.size())
    {
      throw runtime_error("Number of masks must be the same as images.");
    }

    indices = img_indices.empty() ? vector<int>(largest_cam_idx):img_indices;
    if(img_indices.empty())
    {
      std::iota(indices.begin(), indices.end(), 0);
      for(const auto &i: indices)
      {
        if(haveMasks)
        {
          imageMap.emplace(i, make_pair(imgs.at(i), masks.at(i)));
        }
        else
        {
          imageMap.emplace(i, make_pair(imgs.at(i), cv::Mat()));
        }
      }
    }
    else
    {
      if(imgs.size() != indices.size())
      {
        throw runtime_error("Number of indices must be the same as images.");
      }
      std::vector<pair<int, int>> idx_tmp(largest_cam_idx);
      for (int i = 0; i < largest_cam_idx; i++)
      {
        idx_tmp.emplace_back(make_pair(i, indices[i]));
      }
      stable_sort(idx_tmp.begin(), idx_tmp.end(), [](const pair<int, int> &i1, const pair<int, int> &i2) {return i1.second < i2.second;});
      indices.clear();
      for(const auto &i: idx_tmp)
      {
        indices.emplace_back(i.second);
        if(haveMasks)
        {
          imageMap.emplace(i.second, make_pair(imgs.at(i.first), masks.at(i.first)));
        }
        else
        {
          imageMap.emplace(i.second, make_pair(imgs.at(i.first), cv::Mat()));
        }
      }
    }
  }

  Matching::~Matching()
  {
      cout << "Matching::~Matching()" << endl;
  }

  void Matching::loadImages()
  {
    auto startTime = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    const unsigned imageCount = indices.size();
    const unsigned threadCount = std::min(imageCount, std::max(std::thread::hardware_concurrency() / 2u, 1u));
    const unsigned batchSize = std::ceil(imageCount / static_cast<float>(threadCount));

    for (unsigned int i = 0; i < threadCount; ++i)
    {
        const int startIdx = i * batchSize;
        const int endIdx = std::min((i + 1u) * batchSize, imageCount);
        threads.push_back(std::thread(std::bind(&Matching::loadImageThreadFunc, this, startIdx, endIdx)));
    }

    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
    cout << "Loading images took " << timeDiff / 1e3 << " seconds." << endl;
  }

  void Matching::loadImageThreadFunc(const int startIdx, const int endIdx)
  {
    for (auto i = startIdx; i < endIdx; ++i)
    {
        const int &idx = indices[i];
        cv::Mat img1 = cv::imread(img_mask_names.at(idx).first, cv::IMREAD_GRAYSCALE);
        if (img1.data == NULL)
        {
            throw runtime_error("Unable to read image " + img_mask_names.at(idx).first);
        }
        if (!nearZero(img_scaling - 1.0))
        {
            cv::Size imgSi = img1.size();
            cv::Size newImgSi(static_cast<int>(std::round(img_scaling * static_cast<double>(imgSi.width))), static_cast<int>(std::round(img_scaling * static_cast<double>(imgSi.height))));
            cv::Mat img1o = img1;
            cv::resize(img1o, img1, newImgSi, 0, 0, cv::INTER_AREA);
        }
        if (equalizeImgs_){
#if USE_CLAHE_FOR_HIST_EQU
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(16,16));
            clahe->apply(img1, imageMap[idx].first);
#else
            cv::equalizeHist(img1, imageMap[idx].first);
#endif
        }
        else
        {
            img1.copyTo(imageMap[idx].first);
        }
        if(haveMasks)
        {
          cv::Mat img2 = cv::imread(img_mask_names.at(idx).second, cv::IMREAD_GRAYSCALE);
          if (img2.data == NULL)
          {
              throw runtime_error("Unable to read mask image " + img_mask_names.at(idx).second);
          }
          if (!nearZero(img_scaling - 1.0))
          {
              cv::Size imgSi = img2.size();
              cv::Size newImgSi(static_cast<int>(std::round(img_scaling * static_cast<double>(imgSi.width))), static_cast<int>(std::round(img_scaling * static_cast<double>(imgSi.height))));
              cv::Mat img2o = img2;
              cv::resize(img2o, img2, newImgSi, 0, 0, cv::INTER_AREA);
          }
          if (equalizeImgs_){
              cv::equalizeHist(img2, imageMap[idx].second);
          }
          else
          {
              img2.copyTo(imageMap[idx].second);
          }
        }
    }
  }

  bool Matching::compute(const bool affineInvariant, const bool useCuda)
  {
      if (largest_cam_idx < 2)
      {
          std::cerr << "Too less images." << endl;
          return false;
      }
      if (affineInvariant)
      {
          affineInvariantUsed = affineInvariant;
          if (!getFeaturesAffine(useCuda))
          {
              return false;
          }
      }
      else
      {
          bool equ_KpDescr = false;
          affineInvariantUsed = false;
          if (keypoint_types_.size() == 1)
          {
              equ_KpDescr = (keypoint_types_.at(0).compare(descriptor_type_) == 0);
          }
          if (equ_KpDescr){
              if(!getFeatures(useCuda)){
                  return false;
              }
          }else{
              if(!computeKeypointsOnly(useCuda)){
                  return false;
              }
              const bool useCuda_ = (descriptor_type_.compare("ORB") == 0) ? false : useCuda;
              if (!getDescriptors(useCuda_))
              {
                  return false;
              }
#if DBG_SHOW_KEYPOINTS == 2
              if (keypoint_types_.size() > 1){
                  visualizeKeypoints(init_keypoints, imageMap, config_data_ptr, descriptor_type_);
              }
#endif
          }
      }
#if DBG_SHOW_KEYPOINTS
      string kp_str;
      for (const auto &kpt: keypoint_types_){
          kp_str += kpt + "_";
      }
      kp_str += descriptor_type_;
      visualizeKeypoints(keypoints_descriptors, imageMap, config_data_ptr, kp_str);
#endif
      generateMotionMasks();
      findAdditionalKeypointsLessTexture();
      return getMatches(affineInvariant);
  }

  int Matching::getMaxNrFeatures(const bool affineInvariant)
  {
      if (affineInvariant)
      {
          return nrFeaturesToExtract/3;
      }
      bool equ_KpDescr = false;
      if (keypoint_types_.size() == 1)
      {
          equ_KpDescr = (keypoint_types_.at(0).compare(descriptor_type_) == 0);
      }
      if (equ_KpDescr)
      {
          return nrFeaturesToExtract;
      }
      return 4 * nrFeaturesToExtract / 5;
  }

  bool Matching::computeFeaturesOtherImgs(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                                          const std::vector<int> &indices_,
                                          std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                                          std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> *init_keypoints_, 
                                          std::unordered_map<int, std::vector<cv::KeyPoint>> *keypoints_combined_, 
                                          const bool useCuda) 
  {
      int nrKeyPointsMax = getMaxNrFeatures(affineInvariantUsed);
      return computeFeaturesOtherImgs(imageMap_, indices_, keypoints_descriptors_, init_keypoints_, keypoints_combined_, nrKeyPointsMax, useCuda);
  }

  bool Matching::computeFeaturesOtherImgs(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                                          const std::vector<int> &indices_, 
                                          std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                                          std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> *init_keypoints_, 
                                          std::unordered_map<int, std::vector<cv::KeyPoint>> *keypoints_combined_, const int &nrKeyPointsMax, const bool useCuda)
  {
      if (affineInvariantUsed)
      {
          if (!getFeaturesAffine(imageMap_, indices_, keypoints_descriptors_, init_keypoints_, keypoints_combined_, nrKeyPointsMax, useCuda))
          {
              return false;
          }
      }
      else
      {
          bool equ_KpDescr = false;
          if (keypoint_types_.size() == 1)
          {
              equ_KpDescr = (keypoint_types_.at(0).compare(descriptor_type_) == 0);
          }
          if (equ_KpDescr)
          {
              if (!getFeatures(imageMap_, indices_, keypoints_descriptors_, init_keypoints_, keypoints_combined_, nrKeyPointsMax, useCuda))
              {
                  return false;
              }
          }
          else
          {
              std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> init_keypoints_tmp, *init_keypoints_tmp_ptr = nullptr;
              std::unordered_map<int, std::vector<cv::KeyPoint>> keypoints_combined_tmp, *keypoints_combined_tmp_ptr = nullptr;
              if (init_keypoints_)
              {
                  init_keypoints_tmp_ptr = init_keypoints_;
              }
              else
              {
                  init_keypoints_tmp_ptr = &init_keypoints_tmp;
              }
              if (keypoints_combined_)
              {
                  keypoints_combined_tmp_ptr = keypoints_combined_;
              }
              else
              {
                  keypoints_combined_tmp_ptr = &keypoints_combined_tmp;
              }
              if (!computeKeypointsOnly(imageMap_, indices_, *init_keypoints_tmp_ptr, *keypoints_combined_tmp_ptr, nrKeyPointsMax, useCuda))
              {
                  return false;
              }
              const bool useCuda_ = (descriptor_type_.compare("ORB") == 0) ? false : useCuda;
              if (!getDescriptors(imageMap_, *keypoints_combined_tmp_ptr, indices_, keypoints_descriptors_, nrKeyPointsMax, useCuda_))
              {
                  return false;
              }
#if DBG_SHOW_KEYPOINTS == 2 && DBG_SHOW_ADDITIONAL_KP_GEN != 2
              if (keypoint_types_.size() > 1)
              {
                  visualizeKeypoints(*init_keypoints_tmp_ptr, imageMap_, config_data_ptr, descriptor_type_);
              }
#endif
          }
      }
#if DBG_SHOW_KEYPOINTS
      string kp_str = "more_kp_";
      for (const auto &kpt : keypoint_types_)
      {
          kp_str += kpt + "_";
      }
      kp_str += descriptor_type_;
      visualizeKeypoints(keypoints_descriptors_, imageMap_, config_data_ptr, kp_str);
#endif
      return true;
  }

  bool Matching::computeKeypointsOnly(const bool useCuda)
  {
      if(!getKeypoints(useCuda)){
          return false;
      }
      if(!combineKeypoints()){
          return false;
      }
      return true;
  }

  bool Matching::computeKeypointsOnly(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                                      const std::vector<int> &indices_, 
                                      std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &init_keypoints_, 
                                      std::unordered_map<int, std::vector<cv::KeyPoint>> &keypoints_combined_, 
                                      const int &nrKeyPointsMax, 
                                      const bool useCuda)
  {
      if (!getKeypoints(imageMap_, indices_, init_keypoints_, nrKeyPointsMax, useCuda))
      {
          return false;
      }
      if (!combineKeypoints(init_keypoints_, indices_, keypoints_combined_))
      {
          return false;
      }
      return true;
  }

  bool Matching::getMatches()
  {
      return getMatches(false);
  }

  bool Matching::getKeypoints(const bool useCuda) 
  {
      indices_kp.clear();
      for (auto &kpt : keypoint_types_)
      {
          init_keypoints[kpt] = std::unordered_map<int, std::vector<cv::KeyPoint>>();
          for (auto &idx : indices)
          {
              indices_kp.emplace_back(make_pair(kpt, idx));

              init_keypoints[kpt][idx] = std::vector<cv::KeyPoint>();
          }
      }

      int nrKeyPointsMax = getMaxNrFeatures(false);

      auto startTime = std::chrono::steady_clock::now();

      bool ret = getKeypoints(imageMap, indices, init_keypoints, nrKeyPointsMax, useCuda);
      if(!ret){
          return false;
      }

      kp_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
      cout << "Detecting keypoints took " << kp_time_ms / 1e3 << " seconds." << endl;
      return true;
  }

  bool Matching::getKeypoints(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                              const std::vector<int> &indices_, 
                              std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &init_keypoints_, 
                              const int &nrKeyPointsMax, 
                              const bool useCuda)
  {
      assert(!indices_.empty());
      if(init_keypoints_.empty()){
          for (auto &kpt : keypoint_types_)
          {
              init_keypoints_[kpt] = std::unordered_map<int, std::vector<cv::KeyPoint>>();
              for (auto &idx : indices_)
              {
                  init_keypoints_[kpt][idx] = std::vector<cv::KeyPoint>();
              }
          }
      }
      std::vector<std::pair<std::string, int>> indices_kp_cpu, indices_kp_gpu, indices_akaze_gpu;
      for (auto &kpt : keypoint_types_)
      {
          // Disable ORB Cuda (use CPU version instead), as keypoints are only extracted approx. on half the image height
          if (useCuda && IsFeatureCudaTypeSupported(kpt) && kpt.compare("AKAZE") && kpt.compare("ORB"))
          {
              for (auto &idx : indices_)
              {
#ifdef WITH_AKAZE_CUDA
                  indices_kp_gpu.emplace_back(make_pair(kpt, idx));
#else
                  indices_kp_cpu.emplace_back(make_pair(kpt, idx));
#endif
              }
          }
#ifdef WITH_AKAZE_CUDA
          else if (useCuda && !kpt.compare("AKAZE"))
          {
              for (auto &idx : indices_)
              {
                  indices_akaze_gpu.emplace_back(make_pair(kpt, idx));
              }
          }
#endif
          else
          {
              for (auto &idx : indices_)
              {
                  indices_kp_cpu.emplace_back(make_pair(kpt, idx));
              }
          }
      }

      std::vector<std::thread> threads;
      const unsigned extractionsCount = indices_kp_cpu.empty() ? (indices_kp_gpu.empty() ? indices_akaze_gpu.size() : indices_kp_gpu.size()) : indices_kp_cpu.size();
      unsigned threadCount = std::min(extractionsCount, static_cast<unsigned>(cpuCnt));
      unsigned batchSize_cpu = std::ceil(extractionsCount / static_cast<float>(threadCount));
      getThreadBatchSize(extractionsCount, threadCount, batchSize_cpu);

      unsigned exeptCount = 0;
      unsigned threadCount_gpu = 0;
      unsigned threadCount_akaze = 0;
#ifdef WITH_AKAZE_CUDA
      vector<cv::Ptr<matchinglib::cuda::Feature2DAsync>> detector_ptrs_gpu_akaze;
#endif
      if (!indices_kp_cpu.empty()){
          exeptCount = threadCount;
      }
      if (!indices_kp_gpu.empty())
      {
#ifdef WITH_AKAZE_CUDA
          const size_t byteMultiplier_cv = matchinglib::cuda::getDefaultByteMultiplierGpu();
          const size_t byteAdd_cv = matchinglib::cuda::getDefaultByteAddGpu();
          unsigned maxParallel = static_cast<unsigned>(std::max(matchinglib::cuda::getMaxNrGpuThreadsFromMemoryUsage(imageMap_.begin()->second.first, byteMultiplier_cv, byteAdd_cv), 1));
          threadCount_gpu = min(threadCount, maxParallel);
          exeptCount += threadCount_gpu;
#else
          throw runtime_error("Was compiled without CUDA.");
#endif
#ifdef WITH_AKAZE_CUDA
          if (!indices_akaze_gpu.empty())
          {
              cv::Size imgSi = imageMap_.begin()->second.first.size();
              detector_ptrs_gpu_akaze.emplace_back(matchinglib::cuda::AKAZE::create(imgSi.width, imgSi.height));
              detector_ptrs_gpu_akaze.back().dynamicCast<matchinglib::cuda::AKAZE>()->setMaxNrKeypoints(nrKeyPointsMax);
              const size_t byteMultiplier_akaze = detector_ptrs_gpu_akaze.back()->getByteMultiplierGPU();
              const size_t byteAdd_akaze = detector_ptrs_gpu_akaze.back()->getByteAddGPU();
              const size_t nrImgElems = imageMap_.begin()->second.first.total();
              size_t byteReserve = threadCount_gpu * (byteMultiplier_cv * nrImgElems + byteAdd_cv);
              unsigned maxParallel_akaze = static_cast<unsigned>(matchinglib::cuda::getMaxNrGpuThreadsFromMemoryUsage(imageMap_.begin()->second.first, byteMultiplier_akaze, byteAdd_akaze, byteReserve));
              if (!maxParallel_akaze){
                  while (!maxParallel_akaze && threadCount_gpu > 1)
                  {
                      threadCount_gpu--;
                      exeptCount--;
                      byteReserve = threadCount_gpu * (byteMultiplier_cv * nrImgElems + byteAdd_cv);
                      maxParallel_akaze = static_cast<unsigned>(matchinglib::cuda::getMaxNrGpuThreadsFromMemoryUsage(imageMap_.begin()->second.first, byteMultiplier_akaze, byteAdd_akaze, byteReserve));
                  }
                  maxParallel_akaze = 1;
              }
              threadCount_akaze = min(threadCount, maxParallel_akaze);
              exeptCount += threadCount_akaze;
          }
#endif
      }
#ifdef WITH_AKAZE_CUDA
      else if (!indices_akaze_gpu.empty()){
          cv::Size imgSi = imageMap_.begin()->second.first.size();
          detector_ptrs_gpu_akaze.emplace_back(matchinglib::cuda::AKAZE::create(imgSi.width, imgSi.height));
          detector_ptrs_gpu_akaze.back().dynamicCast<matchinglib::cuda::AKAZE>()->setMaxNrKeypoints(nrKeyPointsMax);
          const size_t byteMultiplier_akaze = detector_ptrs_gpu_akaze.back()->getByteMultiplierGPU();
          const size_t byteAdd_akaze = detector_ptrs_gpu_akaze.back()->getByteAddGPU();
          unsigned maxParallel_akaze = static_cast<unsigned>(std::max(matchinglib::cuda::getMaxNrGpuThreadsFromMemoryUsage(imageMap_.begin()->second.first, byteMultiplier_akaze, byteAdd_akaze), 1));
          threadCount_akaze = min(threadCount, maxParallel_akaze);
          exeptCount += threadCount_akaze;
      }
#endif

      const unsigned extractionsCount_gpu = indices_kp_gpu.size();
      unsigned batchSize_gpu = threadCount_gpu > 0 ? std::ceil(extractionsCount_gpu / static_cast<float>(threadCount_gpu)) : 0;
      const unsigned threadCount_gpu_sv = threadCount_gpu;
      getThreadBatchSize(extractionsCount_gpu, threadCount_gpu, batchSize_gpu);
      if (threadCount_gpu_sv != threadCount_gpu){
          exeptCount -= threadCount_gpu_sv - threadCount_gpu;
      }

      const unsigned extractionsCount_akaze = indices_akaze_gpu.size();
      unsigned batchSize_akaze = threadCount_akaze > 0 ? std::ceil(extractionsCount_akaze / static_cast<float>(threadCount_akaze)) : 0;
      const unsigned threadCount_akaze_sv = threadCount_akaze;
      getThreadBatchSize(extractionsCount_akaze, threadCount_akaze, batchSize_akaze);
      if (threadCount_akaze_sv != threadCount_akaze)
      {
          exeptCount -= threadCount_akaze_sv - threadCount_akaze;
      }

      std::vector<std::exception_ptr> thread_exceptions(exeptCount, nullptr);

      vector<std::shared_ptr<std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>>>> kps_thread;
      for (unsigned int i = 0; i < exeptCount; ++i)
      {
          kps_thread.emplace_back(std::make_shared<std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>>>());
      }

      vector<cv::Ptr<cv::FeatureDetector>> detector_ptrs_cpu;
      if (!indices_kp_cpu.empty()){
          for (unsigned int i = 0; i < threadCount; ++i)
          {
              detector_ptrs_cpu.emplace_back(cv::Ptr<cv::FeatureDetector>());
          }

          for (unsigned int i = 0; i < threadCount; ++i)
          {
              const int startIdx = i * batchSize_cpu;
              const int endIdx = std::min((i + 1u) * batchSize_cpu, extractionsCount);
              threads.push_back(std::thread(std::bind(&getKeypointsThreadFunc, startIdx, endIdx, indices_kp_cpu, &imageMap_, detector_ptrs_cpu.at(i), kps_thread.at(i), std::ref(thread_exceptions.at(i)), nrKeyPointsMax)));
          }
      }

#ifdef WITH_AKAZE_CUDA
      vector<cv::Ptr<cv::cuda::Feature2DAsync>> detector_ptrs_gpu;
#endif
      const unsigned int add_cpu = indices_kp_cpu.empty() ? 0 : threadCount;
      if (!indices_kp_gpu.empty())
      {
#ifdef WITH_AKAZE_CUDA
          for (unsigned int i = 0; i < threadCount_gpu; ++i)
          {
              const int startIdx = i * batchSize_gpu;
              if (startIdx >= static_cast<int>(indices_kp_gpu.size()))
              {
                  continue;
              }
              detector_ptrs_gpu.emplace_back(createCudaDetector(get<0>(indices_kp_gpu.at(startIdx)), nrKeyPointsMax));
          }

          for (unsigned int i = 0; i < threadCount_gpu; ++i)
          {
              const int startIdx = i * batchSize_gpu;
              if (startIdx >= static_cast<int>(indices_kp_gpu.size()))
              {
                  continue;
              }
              const int endIdx = std::min((i + 1u) * batchSize_gpu, extractionsCount_gpu);
              threads.push_back(std::thread(std::bind(&getKeypointsCudaThreadFunc, startIdx, endIdx, indices_kp_gpu, &imageMap_, detector_ptrs_gpu.at(i), kps_thread.at(add_cpu + i), std::ref(thread_exceptions.at(add_cpu + i)))));
          }
#else
          throw runtime_error("Was compiled without CUDA.");
#endif
      }

      const unsigned int add_cpu_gpu = indices_kp_gpu.empty() ? add_cpu : add_cpu + threadCount_gpu;
#ifdef WITH_AKAZE_CUDA
      if(!indices_akaze_gpu.empty()){
          std::mt19937 &mt = RandomGenerator::getInstance(std::seed_seq(config_data_ptr->id.begin(), config_data_ptr->id.end())).getTwisterEngineRef();
          cv::Size imgSi = imageMap_.begin()->second.first.size();
          for (unsigned int i = 1; i < threadCount_akaze; ++i)
          {
              const int startIdx = i * batchSize_akaze;
              if (startIdx >= static_cast<int>(indices_akaze_gpu.size()))
              {
                  continue;
              }
              detector_ptrs_gpu_akaze.emplace_back(matchinglib::cuda::AKAZE::create(imgSi.width, imgSi.height));
              detector_ptrs_gpu_akaze.back().dynamicCast<matchinglib::cuda::AKAZE>()->setMaxNrKeypoints(nrKeyPointsMax);
          }

          for (unsigned int i = 0; i < threadCount_akaze; ++i)
          {
              const int startIdx = i * batchSize_akaze;
              if (startIdx >= static_cast<int>(indices_akaze_gpu.size()))
              {
                  continue;
              }
              const int endIdx = std::min((i + 1u) * batchSize_akaze, extractionsCount_akaze);
              threads.push_back(std::thread(std::bind(&getKeypointsAkazeCudaThreadFunc, startIdx, endIdx, indices_akaze_gpu, &imageMap_, detector_ptrs_gpu_akaze.at(i), kps_thread.at(add_cpu_gpu + i), std::ref(thread_exceptions.at(add_cpu_gpu + i)), mt)));
          }
      }
#endif

      for (auto &t : threads)
      {
          if (t.joinable())
          {
              t.join();
          }
      }

      size_t err_cnt = 0;
      for (auto &e : thread_exceptions)
      {
          try{
              if (e)
              {
                  std::rethrow_exception(e);
              }
          }
          catch (const std::exception &e)
          {
              cerr << "Exception during keypoint detection using ";
              if (!indices_kp_cpu.empty() && err_cnt < threadCount)
              {
                  cerr << "the CPU (Thread " << err_cnt << "): ";
              }
              else if (!indices_akaze_gpu.empty() && err_cnt >= add_cpu_gpu)
              {
                  cerr << "AKAZE on the GPU (Thread " << err_cnt << "): ";
              }
              else
              {
                  cerr << "the GPU (Thread " << err_cnt << "): ";
              }
              cerr << e.what() << endl;
              err_cnt++;
          }
      }

      bool check_nr_kp = false;
      if (err_cnt)
      {
          if (keypoint_types_.size() == 1 && err_cnt >= kps_thread.size() / 3)
          {
              return false;
          }
          else if (keypoint_types_.size() > 1){
              check_nr_kp = true;
          }
      }

      bool kp_cpu_valid = false;
      if (!indices_kp_cpu.empty())
      {
          size_t nr_invalids = 0;
          for (unsigned int i = 0; i < threadCount; ++i)
          {
              const int startIdx = i * batchSize_cpu;
              const int endIdx = std::min((i + 1u) * batchSize_cpu, extractionsCount);
              for (auto j = startIdx; j < endIdx; ++j)
              {
                  const std::pair<std::string, int> &idx = indices_kp_cpu.at(j);
                  const string kp_type = idx.first;
                  const int img_idx = idx.second;
                  if (kps_thread.at(i)->find(kp_type) == kps_thread.at(i)->end())
                  {
                      nr_invalids += endIdx - startIdx;
                      break;
                  }
                  if (kps_thread.at(i)->at(kp_type).find(img_idx) == kps_thread.at(i)->at(kp_type).end())
                  {
                      nr_invalids++;
                      continue;
                  }
                  init_keypoints_.at(kp_type).at(img_idx) = kps_thread.at(i)->at(kp_type).at(img_idx);
                  if (check_nr_kp && init_keypoints_.at(kp_type).at(img_idx).size() < 80){
                      nr_invalids++;
                  }
              }
          }
          if (nr_invalids < extractionsCount / 3){
              kp_cpu_valid = true;
          }
      }
      bool kp_gpu_valid = false;
      if (!indices_kp_gpu.empty())
      {
          size_t nr_invalids = 0;
          for (unsigned int i = 0; i < threadCount_gpu; ++i)
          {
              const int startIdx = i * batchSize_gpu;
              if (startIdx >= static_cast<int>(indices_kp_gpu.size()))
              {
                  continue;
              }
              const int endIdx = std::min((i + 1u) * batchSize_gpu, extractionsCount_gpu);
              for (auto j = startIdx; j < endIdx; ++j)
              {
                  const std::pair<std::string, int> &idx = indices_kp_gpu.at(j);
                  const string kp_type = idx.first;
                  const int img_idx = idx.second;
                  if (kps_thread.at(add_cpu + i)->find(kp_type) == kps_thread.at(add_cpu + i)->end())
                  {
                      nr_invalids += endIdx - startIdx;
                      break;
                  }
                  if (kps_thread.at(add_cpu + i)->at(kp_type).find(img_idx) == kps_thread.at(add_cpu + i)->at(kp_type).end())
                  {
                      nr_invalids++;
                      continue;
                  }
                  init_keypoints_.at(kp_type).at(img_idx) = kps_thread.at(add_cpu + i)->at(kp_type).at(img_idx);
                  // cout << "GPU: " << kps_thread.at(add_cpu + i)->at(kp_type).at(img_idx).size() << endl;
                  if (check_nr_kp && init_keypoints_.at(kp_type).at(img_idx).size() < 80)
                  {
                      nr_invalids++;
                  }
              }
          }
          if (nr_invalids < extractionsCount_gpu / 3)
          {
              kp_gpu_valid = true;
          }
      }
      bool kp_akaze_valid = false;
      if (!indices_akaze_gpu.empty())
      {
          size_t nr_invalids = 0;
          for (unsigned int i = 0; i < threadCount_akaze; ++i)
          {
              const int startIdx = i * batchSize_akaze;
              if (startIdx >= static_cast<int>(indices_akaze_gpu.size()))
              {
                  continue;
              }
              const int endIdx = std::min((i + 1u) * batchSize_akaze, extractionsCount_akaze);
              for (auto j = startIdx; j < endIdx; ++j)
              {
                  const std::pair<std::string, int> &idx = indices_akaze_gpu.at(j);
                  const string kp_type = idx.first;
                  const int img_idx = idx.second;
                  if (kps_thread.at(add_cpu_gpu + i)->find(kp_type) == kps_thread.at(add_cpu_gpu + i)->end())
                  {
                      nr_invalids += endIdx - startIdx;
                      break;
                  }
                  if (kps_thread.at(add_cpu_gpu + i)->at(kp_type).find(img_idx) == kps_thread.at(add_cpu_gpu + i)->at(kp_type).end())
                  {
                      nr_invalids++;
                      continue;
                  }
                  init_keypoints_.at(kp_type).at(img_idx) = kps_thread.at(add_cpu_gpu + i)->at(kp_type).at(img_idx);
                  // cout << "AKAZE: " << kps_thread.at(add_cpu_gpu + i)->at(kp_type).at(img_idx).size() << endl;
                  if (check_nr_kp && init_keypoints_.at(kp_type).at(img_idx).size() < 80)
                  {
                      nr_invalids++;
                  }
              }
          }
          if (nr_invalids < extractionsCount_akaze / 3)
          {
              kp_akaze_valid = true;
          }
      }
      
      return kp_cpu_valid || kp_gpu_valid || kp_akaze_valid;
  }

  void getKeypointsThreadFunc(const int startIdx, 
                              const int endIdx, 
                              const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                              const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                              cv::Ptr<cv::FeatureDetector> detector, 
                              std::shared_ptr<std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>>> keypoints, 
                              std::exception_ptr &thread_exception, 
                              const int limitNrFeatures)
  {
      try{
          size_t nrInvalids = 0;
          for (auto i = startIdx; i < endIdx; ++i)
          {
              const std::pair<std::string, int> idx = indices_kp_threads[i];
              const string kp_type = idx.first;
              const int img_idx = idx.second;
              if (keypoints->find(kp_type) == keypoints->end()){
                  keypoints->emplace(kp_type, std::unordered_map<int, std::vector<cv::KeyPoint>>());
              }
              keypoints->at(kp_type).emplace(img_idx, std::vector<cv::KeyPoint>());
              std::vector<cv::KeyPoint> &keypoints_tmp = keypoints->at(kp_type).at(img_idx);
              int err = matchinglib::getKeypoints(imageMap_threads->at(img_idx).first, keypoints_tmp, kp_type, detector, true, limitNrFeatures);
              if (err)
              {
                  string msg = "Unable to detect " + kp_type + " keypoints on image " + std::to_string(img_idx);
                  cerr << msg << endl;
                  nrInvalids++;
                  if (nrInvalids > 5)
                  {
                      throw runtime_error("Detection of keypoints using the CPU failed for more than 5 images.");
                  }
              }
          }
          if (nrInvalids)
          {
              throw runtime_error("Detection of keypoints using the CPU failed for one or more images.");
          }
      }
      catch (...)
      {
          thread_exception = std::current_exception();
          return;
      }
  }

#ifdef WITH_AKAZE_CUDA
  void getKeypointsCudaThreadFunc(const int startIdx, 
                                  const int endIdx, 
                                  const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                                  const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                  cv::Ptr<cv::cuda::Feature2DAsync> detector, 
                                  std::shared_ptr<std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>>> keypoints, 
                                  std::exception_ptr &thread_exception)
  {
      try
      {
          cv::cuda::Stream stream;
          size_t nrInvalids = 0;
          for (auto i = startIdx; i < endIdx; ++i)
          {
              const std::pair<std::string, int> idx = indices_kp_threads[i];
              const string kp_type = idx.first;
              const int img_idx = idx.second;
              if (keypoints->find(kp_type) == keypoints->end())
              {
                  keypoints->emplace(kp_type, std::unordered_map<int, std::vector<cv::KeyPoint>>());
              }
              keypoints->at(kp_type).emplace(img_idx, std::vector<cv::KeyPoint>());
              std::vector<cv::KeyPoint> &keypoints_tmp = keypoints->at(kp_type).at(img_idx);
              cv::cuda::GpuMat imgGpu, keypoints_gpu;
              imgGpu.upload(imageMap_threads->at(img_idx).first, stream);
              detector->detectAndComputeAsync(imgGpu, cv::noArray(), keypoints_gpu, cv::noArray(), false, stream);
              detector->convert(keypoints_gpu, keypoints_tmp);
              if (keypoints_tmp.empty())
              {
                  string msg = "Unable to detect " + kp_type + " keypoints on image " + std::to_string(img_idx) + " using the GPU.";
                  cerr << msg << endl;
                  nrInvalids++;
                  if (nrInvalids > 5)
                  {
                      throw runtime_error("Detection of CV keypoints using the GPU failed for more than 5 images.");
                  }
              }
              // cout << "Nr " << kp_type << " keypoints on image " << img_idx.first << "-" << img_idx.second << ": " << keypoints_tmp.size() << endl;
          }
          if (nrInvalids)
          {
              throw runtime_error("Detection of CV keypoints using the GPU failed for one or more images.");
          }
      }
      catch (...)
      {
          thread_exception = std::current_exception();
          return;
      }
  }
#endif

  bool Matching::combineKeypoints()
  {
      return combineKeypoints(init_keypoints, indices, keypoints_combined);
  }

  bool Matching::combineKeypoints(std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &init_keypoints_, 
                                  const std::vector<int> &indices_, 
                                  std::unordered_map<int, std::vector<cv::KeyPoint>> &keypoints_combined_)
  {
      if (nr_keypoint_types == 1){
          keypoints_combined_ = init_keypoints_.at(keypoint_types_.at(0));
          return true;
      }

      //Get max response for every keypoint type and combine keypoint for every image seperately
      std::unordered_map<std::string, float> resp;
      std::unordered_map<int, KeypointSearchSimple> keypoints_combinedMaps;
      for(const auto &kpt : keypoint_types_){
          resp.emplace(kpt, 0);
      }
      for(const auto &ci: indices_){
          keypoints_combinedMaps.emplace(ci, KeypointSearchSimple());
      }
      for(const auto &kpt: init_keypoints_){
          float &max_resp = resp.at(kpt.first);
          for(const auto &kpi: kpt.second){
              for (const auto &kp : kpi.second)
              {
                  if (kp.response > max_resp){
                      max_resp = kp.response;
                  }
              }
          }
      }
      for (auto &kpt : init_keypoints_)
      {
          const float &max_resp = resp.at(kpt.first);
          for (auto &kpi : kpt.second)
          {
              KeypointSearchSimple &ks = keypoints_combinedMaps.at(kpi.first);
              for (auto &kp : kpi.second)
              {
                  kp.response /= max_resp;
                  if (kp.octave < 0 || kp.octave > 20)
                  {
                      kp.octave = 0;
                  }
                  ks.add(kp);
              }
          }
      }
      for (const auto &kpc : keypoints_combinedMaps)
      {
          if(kpc.second.empty()){
              return false;
          }
          keypoints_combined_.emplace(kpc.first, kpc.second.composeAll());
      }
      // for (const auto &kpty : keypoints_combined_)
      // {
      //     cout << "Nr kp (" << kpty.first.first << "-" << kpty.first.second << "): " << kpty.second.size() << endl;
      // }
      return true;
  }

  bool Matching::getDescriptors(const bool useCuda) 
  {
      auto startTime = std::chrono::steady_clock::now();

      bool ret = getDescriptors(imageMap, keypoints_combined, indices, keypoints_descriptors, nrFeaturesToExtract, useCuda);

      descr_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
      cout << "Calculating descriptors took " << descr_time_ms / 1e3 << " seconds." << endl;
      return ret;
  }

  bool Matching::getDescriptors(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                                const std::unordered_map<int, std::vector<cv::KeyPoint>> &keypoints_combined_, 
                                const std::vector<int> &indices_, 
                                std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                                const int &nrKeyPointsMax, 
                                const bool useCuda)
  {
      for (const auto &idx : indices_)
      {
          keypoints_descriptors_.emplace(idx, make_pair(keypoints_combined_.at(idx), cv::Mat()));
      }
#ifdef WITH_AKAZE_CUDA
      const bool useCudaOpenCV = (useCuda && IsFeatureCudaTypeSupported(descriptor_type_) && descriptor_type_.compare("AKAZE"));
      const bool useAkazeCuda = (useCuda && !useCudaOpenCV && !descriptor_type_.compare("AKAZE"));
#else
      const bool useCudaOpenCV = false;
      const bool useAkazeCuda = false;
#endif

      std::vector<std::thread> threads;
      const unsigned extractionsCount = indices_.size();
      unsigned threadCount = std::min(extractionsCount, static_cast<unsigned>(cpuCnt));
      vector<cv::Ptr<matchinglib::cuda::Feature2DAsync>> detector_ptrs_gpu_akaze;
      if(useCudaOpenCV){
          const size_t byteMultiplier_cv = matchinglib::cuda::getDefaultByteMultiplierGpu();
          const size_t byteAdd_cv = matchinglib::cuda::getDefaultByteAddGpu();
          const unsigned maxParallel = static_cast<unsigned>(std::max(matchinglib::cuda::getMaxNrGpuThreadsFromMemoryUsage(imageMap_.begin()->second.first, byteMultiplier_cv, byteAdd_cv), 1));
          threadCount = min(threadCount, maxParallel);
      }
#ifdef WITH_AKAZE_CUDA
      else if (useAkazeCuda)
      {
          cv::Size imgSi = imageMap_.begin()->second.first.size();
          detector_ptrs_gpu_akaze.emplace_back(matchinglib::cuda::AKAZE::create(imgSi.width, imgSi.height));
          detector_ptrs_gpu_akaze.back().dynamicCast<matchinglib::cuda::AKAZE>()->setMaxNrKeypoints(nr_keypoint_types * nrKeyPointsMax);
          const size_t byteMultiplier_akaze = detector_ptrs_gpu_akaze.back()->getByteMultiplierGPU();
          const size_t byteAdd_akaze = detector_ptrs_gpu_akaze.back()->getByteAddGPU();
          const unsigned maxParallel_akaze = static_cast<unsigned>(std::max(matchinglib::cuda::getMaxNrGpuThreadsFromMemoryUsage(imageMap_.begin()->second.first, byteMultiplier_akaze, byteAdd_akaze), 1));
          threadCount = min(threadCount, maxParallel_akaze);
      }
#endif
      unsigned batchSize = std::ceil(extractionsCount / static_cast<float>(threadCount));
      getThreadBatchSize(extractionsCount, threadCount, batchSize);

      vector<cv::Ptr<cv::DescriptorExtractor>> extractor_ptrs_cpu;
#ifdef WITH_AKAZE_CUDA
      vector<cv::Ptr<cv::cuda::Feature2DAsync>> extractor_ptrs_gpu;
#endif
      vector<std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>>> kps_thread;
      vector<std::shared_ptr<std::unordered_map<int, cv::Mat>>> descr_thread;
      std::vector<std::exception_ptr> thread_exceptions(threadCount, nullptr);
      if (useCudaOpenCV){
#ifdef WITH_AKAZE_CUDA
          for (unsigned int i = 0; i < threadCount; ++i)
          {
              extractor_ptrs_gpu.emplace_back(createCudaExtractor(descriptor_type_, nr_keypoint_types * nrKeyPointsMax));
              const int startIdx = i * batchSize;
              const int endIdx = std::min((i + 1u) * batchSize, extractionsCount);
              kps_thread.emplace_back(std::make_shared<std::unordered_map<int, std::vector<cv::KeyPoint>>>());
              for (auto j = startIdx; j < endIdx; ++j)
              {
                  const int &img_idx = indices_.at(j);
                  kps_thread.back()->emplace(img_idx, keypoints_descriptors_.at(img_idx).first);
              }
              descr_thread.emplace_back(std::make_shared<std::unordered_map<int, cv::Mat>>());
          }

          for (unsigned int i = 0; i < threadCount; ++i)
          {
              const int startIdx = i * batchSize;
              const int endIdx = std::min((i + 1u) * batchSize, extractionsCount);
              threads.push_back(std::thread(std::bind(&getDescriptorsCudaThreadFunc, startIdx, endIdx, descriptor_type_, indices_, &imageMap_, extractor_ptrs_gpu.at(i), kps_thread.at(i), descr_thread.at(i), std::ref(thread_exceptions.at(i)))));
          }
#else
          throw runtime_error("Was compiled without CUDA.");
#endif
      }
#ifdef WITH_AKAZE_CUDA
      else if (useAkazeCuda)
      {
          cv::Size imgSi = imageMap_.begin()->second.first.size();
          std::mt19937 &mt = RandomGenerator::getInstance(std::seed_seq(config_data_ptr->id.begin(), config_data_ptr->id.end())).getTwisterEngineRef();
          for (unsigned int i = 0; i < threadCount; ++i)
          {
              if(i){
                  detector_ptrs_gpu_akaze.emplace_back(matchinglib::cuda::AKAZE::create(imgSi.width, imgSi.height));
                  detector_ptrs_gpu_akaze.back().dynamicCast<matchinglib::cuda::AKAZE>()->setMaxNrKeypoints(nr_keypoint_types * nrKeyPointsMax);
              }
              const int startIdx = i * batchSize;
              const int endIdx = std::min((i + 1u) * batchSize, extractionsCount);
              kps_thread.emplace_back(std::make_shared<std::unordered_map<int, std::vector<cv::KeyPoint>>>());
              for (auto j = startIdx; j < endIdx; ++j)
              {
                  const int &img_idx = indices_.at(j);
                  kps_thread.back()->emplace(img_idx, keypoints_descriptors_.at(img_idx).first);
              }
              descr_thread.emplace_back(std::make_shared<std::unordered_map<int, cv::Mat>>());
          }

          for (unsigned int i = 0; i < threadCount; ++i)
          {
              const int startIdx = i * batchSize;
              const int endIdx = std::min((i + 1u) * batchSize, extractionsCount);
              threads.push_back(std::thread(std::bind(&getDescriptorsAkazeCudaThreadFunc, startIdx, endIdx, indices_, &imageMap_, detector_ptrs_gpu_akaze.at(i), kps_thread.at(i), descr_thread.at(i), std::ref(thread_exceptions.at(i)), mt)));
          }
      }
#endif
      else
      {
          for (unsigned int i = 0; i < threadCount; ++i)
          {
              extractor_ptrs_cpu.emplace_back(cv::Ptr<cv::DescriptorExtractor>());
              const int startIdx = i * batchSize;
              const int endIdx = std::min((i + 1u) * batchSize, extractionsCount);
              kps_thread.emplace_back(std::make_shared<std::unordered_map<int, std::vector<cv::KeyPoint>>>());
              for (auto j = startIdx; j < endIdx; ++j){
                  const int &img_idx = indices_.at(j);
                  kps_thread.back()->emplace(img_idx, keypoints_descriptors_.at(img_idx).first);
              }
              descr_thread.emplace_back(std::make_shared<std::unordered_map<int, cv::Mat>>());
          }

          for (unsigned int i = 0; i < threadCount; ++i)
          {
              const int startIdx = i * batchSize;
              const int endIdx = std::min((i + 1u) * batchSize, extractionsCount);
              threads.push_back(std::thread(std::bind(&getDescriptorsThreadFunc, startIdx, endIdx, descriptor_type_, indices_, &imageMap_, extractor_ptrs_cpu.at(i), kps_thread.at(i), descr_thread.at(i), std::ref(thread_exceptions.at(i)))));
          }
      }

      for (auto &t : threads)
      {
          if (t.joinable())
          {
              t.join();
          }
      }

      size_t err_cnt = 0;
      for (auto &e : thread_exceptions)
      {
          try{
              if (e)
              {
                  std::rethrow_exception(e);
              }
          }
          catch (const std::exception &e)
          {
              cerr << "Exception during descriptor calculation (Thread " << err_cnt << "): " << e.what() << endl;
              err_cnt++;
          }
      }

      if(err_cnt){
          return false;
      }

      for (unsigned int i = 0; i < threadCount; ++i)
      {
          for (const auto &kp_ci : *kps_thread.at(i))
          {
              if (kp_ci.second.empty()){
                  return false;
              }
              keypoints_descriptors_.at(kp_ci.first).first = kp_ci.second;
              keypoints_descriptors_.at(kp_ci.first).second = descr_thread.at(i)->at(kp_ci.first).clone();
              // cout << "Thread " << i << ": " << descr_thread.at(i)->at(kp_ci.first).rows << endl;
          }
      }

      return true;
  }

  void getDescriptorsThreadFunc(const int startIdx, 
                                const int endIdx, 
                                const std::string descr_type, 
                                const std::vector<int> indices_threads, 
                                const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                cv::Ptr<cv::DescriptorExtractor> extractor, 
                                std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors, 
                                std::exception_ptr &thread_exception)
  {
      try{
          for (auto i = startIdx; i < endIdx; ++i)
          {
              const int &img_idx = indices_threads.at(i);
              std::vector<cv::KeyPoint> &keypoints_tmp = keypoints->at(img_idx);
              descriptors->emplace(img_idx, cv::Mat());
              cv::Mat &descr_tmp = descriptors->at(img_idx);
              int err = matchinglib::getDescriptors(imageMap_threads->at(img_idx).first, keypoints_tmp, descr_type, descr_tmp, extractor);
              if (err)
              {
                  string msg = "Unable to calculate " + descr_type + " descriptors of image " + std::to_string(img_idx);
                  throw runtime_error(msg);
              }
          }
      }
      catch (...)
      {
          thread_exception = std::current_exception();
          return;
      }
  }

#ifdef WITH_AKAZE_CUDA
  void getDescriptorsCudaThreadFunc(const int startIdx, 
                                    const int endIdx, 
                                    const std::string descr_type, 
                                    const std::vector<int> indices_threads, 
                                    const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                    cv::Ptr<cv::cuda::Feature2DAsync> extractor, 
                                    std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                    std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors, 
                                    std::exception_ptr &thread_exception)
  {
      try
      {
          for (auto i = startIdx; i < endIdx; ++i)
          {
              const int &img_idx = indices_threads.at(i);
              std::vector<cv::KeyPoint> &keypoints_tmp = keypoints->at(img_idx);
              descriptors->emplace(img_idx, cv::Mat());
              cv::Mat &descr_tmp = descriptors->at(img_idx);
              cv::cuda::GpuMat imgGpu;
              imgGpu.upload(imageMap_threads->at(img_idx).first);
              cv::cuda::GpuMat descriptors_cuda;
              extractor->detectAndCompute(imgGpu, cv::noArray(), keypoints_tmp, descriptors_cuda, true);
              if (descriptors_cuda.empty())
              {
                  string msg = "Unable to calculate " + descr_type + " descriptors of image " + std::to_string(img_idx) + " using the GPU.";
                  throw runtime_error(msg);
              }
              descriptors_cuda.download(descr_tmp);
          }
      }
      catch (...)
      {
          thread_exception = std::current_exception();
          return;
      }
  }
#endif

#ifdef WITH_AKAZE_CUDA
  void getDescriptorsAkazeCudaThreadFunc(const int startIdx, 
                                         const int endIdx, 
                                         const std::vector<int> indices_threads, 
                                         const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                         cv::Ptr<matchinglib::cuda::Feature2DAsync> extractor, 
                                         std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                         std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors, 
                                         std::exception_ptr &thread_exception, 
                                         std::mt19937 mt)
  {
      try
      {
          cv::cuda::Stream stream;
          for (auto i = startIdx; i < endIdx; ++i)
          {
              const int &img_idx = indices_threads.at(i);
              std::vector<cv::KeyPoint> &keypoints_tmp = keypoints->at(img_idx);
              descriptors->emplace(img_idx, cv::Mat());
              cv::Mat &descr_tmp = descriptors->at(img_idx);
              extractor->detectAndComputeAsync(imageMap_threads->at(img_idx).first, cv::noArray(), keypoints_tmp, descr_tmp, mt, true, stream);
              if (descr_tmp.empty())
              {
                  string msg = "Unable to calculate AKAZE descriptors of image " + std::to_string(img_idx) + " using the GPU.";
                  throw runtime_error(msg);
              }
          }
      }
      catch (...)
      {
          thread_exception = std::current_exception();
          return;
      }
  }
#endif

  bool Matching::getFeatures(const bool useCuda)
  {
      std::unordered_map<int, std::vector<cv::KeyPoint>> kps;
      indices_kp.clear();
      for (auto &idx : indices)
      {
          indices_kp.emplace_back(make_pair(descriptor_type_, idx));
          kps.emplace(idx, std::vector<cv::KeyPoint>());
      }
      init_keypoints.clear();
      init_keypoints.emplace(descriptor_type_, move(kps));
      keypoints_combined = init_keypoints.at(descriptor_type_);

      int nrKeyPointsMax = getMaxNrFeatures(false);

      auto startTime = std::chrono::steady_clock::now();

      bool ret = getFeatures(imageMap, indices, indices_kp, keypoints_descriptors, nrKeyPointsMax, useCuda);
      if(!ret){
          return false;
      }

      for (const auto &kpd : keypoints_descriptors)
      {
          init_keypoints.at(descriptor_type_).at(kpd.first) = kpd.second.first;
      }
      keypoints_combined = init_keypoints.at(descriptor_type_);

      kp_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
      cout << "Detecting features took " << kp_time_ms / 1e3 << " seconds." << endl;
      return true;
  }

  bool Matching::getFeatures(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                             const std::vector<int> &indices_, 
                             std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                             std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> *init_keypoints_, 
                             std::unordered_map<int, std::vector<cv::KeyPoint>> *keypoints_combined_, 
                             const int &nrKeyPointsMax, 
                             const bool useCuda)
  {
      std::vector<std::pair<std::string, int>> indices_kp_tmp;
      std::unordered_map<int, std::vector<cv::KeyPoint>> kps;
      for (auto &idx : indices_)
      {
          indices_kp_tmp.emplace_back(make_pair(descriptor_type_, idx));
          kps.emplace(idx, std::vector<cv::KeyPoint>());
      }
      init_keypoints_->emplace(descriptor_type_, move(kps));

      bool ret = getFeatures(imageMap_, indices_, indices_kp_tmp, keypoints_descriptors_, nrKeyPointsMax, useCuda);
      if (!ret)
      {
          return false;
      }

      if (init_keypoints_)
      {
          for (const auto &kpd : keypoints_descriptors_)
          {
              init_keypoints_->at(descriptor_type_).at(kpd.first) = kpd.second.first;
          }
      }
      else if (keypoints_combined_)
      {
          for (const auto &kpd : keypoints_descriptors_)
          {
              keypoints_combined_->emplace(kpd.first, kpd.second.first);
          }
      }
      if (init_keypoints_ && keypoints_combined_)
      {
          *keypoints_combined_ = init_keypoints_->at(descriptor_type_);
      }
      return true;
  }

  bool Matching::getFeatures(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                             const std::vector<int> &indices_, 
                             const std::vector<std::pair<std::string, int>> &indices_kp_, 
                             std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                             const int &nrKeyPointsMax, 
                             const bool useCuda)
  {
      assert(!indices_.empty() && indices_.size() == indices_kp_.size());
      // Disable ORB Cuda (use CPU version instead), as keypoints are only extracted approx. on half the image height
#ifdef WITH_AKAZE_CUDA
      const bool useCudaOpenCV = (useCuda && IsFeatureCudaTypeSupported(descriptor_type_) && descriptor_type_.compare("AKAZE") && descriptor_type_.compare("ORB"));
      const bool useAkazeCuda = (useCuda && !useCudaOpenCV && !descriptor_type_.compare("AKAZE"));
#else
      const bool useCudaOpenCV = false;
      const bool useAkazeCuda = false;
#endif

      keypoints_descriptors_.clear();
      for (auto &idx : indices_)
      {
          keypoints_descriptors_.emplace(idx, make_pair(keypoints_combined.at(idx), cv::Mat()));
      }

      std::vector<std::thread> threads;
      const unsigned extractionsCount = indices_kp_.size();
      unsigned threadCount = std::min(extractionsCount, static_cast<unsigned>(cpuCnt));
#ifdef WITH_AKAZE_CUDA
      vector<cv::Ptr<matchinglib::cuda::Feature2DAsync>> detector_ptrs_gpu_akaze;
#endif
      if (useCudaOpenCV)
      {
#ifdef WITH_AKAZE_CUDA
          const size_t byteMultiplier_cv = matchinglib::cuda::getDefaultByteMultiplierGpu();
          const size_t byteAdd_cv = matchinglib::cuda::getDefaultByteAddGpu();
          const unsigned maxParallel = static_cast<unsigned>(std::max(matchinglib::cuda::getMaxNrGpuThreadsFromMemoryUsage(imageMap_.begin()->second.first, byteMultiplier_cv, byteAdd_cv), 1));
          threadCount = min(threadCount, maxParallel);
#else
          throw runtime_error("Was compiled without CUDA.");
#endif
      }
#ifdef WITH_AKAZE_CUDA
      else if (useAkazeCuda)
      {
          cv::Size imgSi = imageMap_.begin()->second.first.size();
          detector_ptrs_gpu_akaze.emplace_back(matchinglib::cuda::AKAZE::create(imgSi.width, imgSi.height));
          detector_ptrs_gpu_akaze.back().dynamicCast<matchinglib::cuda::AKAZE>()->setMaxNrKeypoints(nrKeyPointsMax);
          const size_t byteMultiplier_akaze = detector_ptrs_gpu_akaze.back()->getByteMultiplierGPU();
          const size_t byteAdd_akaze = detector_ptrs_gpu_akaze.back()->getByteAddGPU();
          const unsigned maxParallel_akaze = static_cast<unsigned>(std::max(matchinglib::cuda::getMaxNrGpuThreadsFromMemoryUsage(imageMap_.begin()->second.first, byteMultiplier_akaze, byteAdd_akaze), 1));
          threadCount = min(threadCount, maxParallel_akaze);
      }
#endif
      unsigned batchSize = std::ceil(extractionsCount / static_cast<float>(threadCount));
      getThreadBatchSize(extractionsCount, threadCount, batchSize);

      vector<cv::Ptr<cv::Feature2D>> kp_detect_ptrs_cpu;
#ifdef WITH_AKAZE_CUDA
      vector<cv::Ptr<cv::cuda::Feature2DAsync>> kp_detect_ptrs_gpu;
#endif
      vector<std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>>> kps_thread;
      vector<std::shared_ptr<std::unordered_map<int, cv::Mat>>> descr_thread;
      for (unsigned int i = 0; i < threadCount; ++i)
      {
          kps_thread.emplace_back(std::make_shared<std::unordered_map<int, std::vector<cv::KeyPoint>>>());
          descr_thread.emplace_back(std::make_shared<std::unordered_map<int, cv::Mat>>());
      }

      if(useCudaOpenCV){
#ifdef WITH_AKAZE_CUDA
          for (unsigned int i = 0; i < threadCount; ++i)
          {
              kp_detect_ptrs_gpu.emplace_back(createCudaExtractor(descriptor_type_, nrKeyPointsMax));
          }

          for (unsigned int i = 0; i < threadCount; ++i)
          {
              const int startIdx = i * batchSize;
              const int endIdx = std::min((i + 1u) * batchSize, extractionsCount);
              threads.push_back(std::thread(std::bind(&getFeaturesCudaThreadFunc, startIdx, endIdx, indices_kp_, &imageMap_, kp_detect_ptrs_gpu.at(i), kps_thread.at(i), descr_thread.at(i))));
          }
#else
          throw runtime_error("Was compiled without CUDA.");
#endif
      }
#ifdef WITH_AKAZE_CUDA
      else if(useAkazeCuda)
      {
          std::mt19937 &mt = RandomGenerator::getInstance(std::seed_seq(config_data_ptr->id.begin(), config_data_ptr->id.end())).getTwisterEngineRef();
          cv::Size imgSi = imageMap_.begin()->second.first.size();
          for (unsigned int i = 1; i < threadCount; ++i)
          {
              detector_ptrs_gpu_akaze.emplace_back(matchinglib::cuda::AKAZE::create(imgSi.width, imgSi.height));
              detector_ptrs_gpu_akaze.back().dynamicCast<matchinglib::cuda::AKAZE>()->setMaxNrKeypoints(nrKeyPointsMax);
          }

          for (unsigned int i = 0; i < threadCount; ++i)
          {
              const int startIdx = i * batchSize;
              const int endIdx = std::min((i + 1u) * batchSize, extractionsCount);
              threads.push_back(std::thread(std::bind(&getFeaturesAkazeCudaThreadFunc, startIdx, endIdx, indices_kp_, &imageMap_, detector_ptrs_gpu_akaze.at(i), kps_thread.at(i), descr_thread.at(i), mt)));
          }
      }
#endif
      else
      {
          for (unsigned int i = 0; i < threadCount; ++i)
          {
              kp_detect_ptrs_cpu.emplace_back(createExtractor(descriptor_type_, descriptor_type_, nrKeyPointsMax));
          }

          for (unsigned int i = 0; i < threadCount; ++i)
          {
              const int startIdx = i * batchSize;
              const int endIdx = std::min((i + 1u) * batchSize, extractionsCount);
              threads.push_back(std::thread(std::bind(&getFeaturesThreadFunc, startIdx, endIdx, indices_kp_, &imageMap_, kp_detect_ptrs_cpu.at(i), kps_thread.at(i), descr_thread.at(i))));
          }
      }

      for (auto &t : threads)
      {
          if (t.joinable())
          {
              t.join();
          }
      }

      for (unsigned int i = 0; i < threadCount; ++i)
      {
          for (const auto &kp_ci : *kps_thread.at(i))
          {
              if(kp_ci.second.empty()){
                  return false;
              }
              keypoints_descriptors_.at(kp_ci.first).first = kp_ci.second;
              keypoints_descriptors_.at(kp_ci.first).second = descr_thread.at(i)->at(kp_ci.first).clone();
          }
      }

      return true;
  }

  void getFeaturesThreadFunc(const int startIdx, 
                             const int endIdx, 
                             const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                             const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                             cv::Ptr<cv::Feature2D> kp_detect_ptr, 
                             std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                             std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors)
  {
      for (auto i = startIdx; i < endIdx; ++i)
      {
          const std::pair<std::string, int> idx = indices_kp_threads.at(i);
          const int img_idx = idx.second;
          descriptors->emplace(img_idx, cv::Mat());
          keypoints->emplace(img_idx, std::vector<cv::KeyPoint>());
          cv::Mat &descr_tmp = descriptors->at(img_idx);
          std::vector<cv::KeyPoint> &kps_tmp = keypoints->at(img_idx);
          kp_detect_ptr->detectAndCompute(imageMap_threads->at(img_idx).first, cv::noArray(), kps_tmp, descr_tmp);
      }
  }

#ifdef WITH_AKAZE_CUDA
  void getFeaturesCudaThreadFunc(const int startIdx, 
                                 const int endIdx, 
                                 const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                                 const std::unordered_map<std::pair<std::string, int>> *imageMap_threads, 
                                 cv::Ptr<cv::cuda::Feature2DAsync> kp_detect_ptr, 
                                 std::shared_ptr<std::unordered_map<std::pair<std::string, int>>> keypoints, 
                                 std::shared_ptr<std::unordered_map<std::pair<std::string, int>>> descriptors)
  {
      cv::cuda::Stream stream;
      for (auto i = startIdx; i < endIdx; ++i)
      {
          const std::pair<std::string, int> idx = indices_kp_threads.at(i);
          const int img_idx = idx.second;
          descriptors->emplace(img_idx, cv::Mat());
          keypoints->emplace(img_idx, std::vector<cv::KeyPoint>());
          cv::Mat &descr_tmp = descriptors->at(img_idx);
          std::vector<cv::KeyPoint> &kps_tmp = keypoints->at(img_idx);
          cv::cuda::GpuMat imgGpu;
          imgGpu.upload(imageMap_threads->at(img_idx).first, stream);
          cv::cuda::GpuMat descriptors_cuda, keypoints_gpu;
          kp_detect_ptr->detectAndComputeAsync(imgGpu, cv::noArray(), keypoints_gpu, descriptors_cuda, false, stream);
          descriptors_cuda.download(descr_tmp, stream);
          kp_detect_ptr->convert(keypoints_gpu, kps_tmp);
      }
  }
#endif

#ifdef WITH_AKAZE_CUDA
  void getFeaturesAkazeCudaThreadFunc(const int startIdx, 
                                      const int endIdx, 
                                      const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                                      const std::unordered_map<std::pair<std::string, int>> *imageMap_threads, 
                                      cv::Ptr<matchinglib::cuda::Feature2DAsync> kp_detect_ptr, 
                                      std::shared_ptr<std::unordered_map<std::pair<std::string, int>>> keypoints, 
                                      std::shared_ptr<std::unordered_map<std::pair<std::string, int>>> descriptors, std::mt19937 mt)
  {
      cv::cuda::Stream stream;
      for (auto i = startIdx; i < endIdx; ++i)
      {
          const std::pair<std::string, int> idx = indices_kp_threads.at(i);
          const int img_idx = idx.second;
          descriptors->emplace(img_idx, cv::Mat());
          keypoints->emplace(img_idx, std::vector<cv::KeyPoint>());
          cv::Mat &descr_tmp = descriptors->at(img_idx);
          std::vector<cv::KeyPoint> &kps_tmp = keypoints->at(img_idx);
          kp_detect_ptr->detectAndComputeAsync(imageMap_threads->at(img_idx).first, cv::noArray(), kps_tmp, descr_tmp, mt, false, stream);
      }
  }
#endif

} // namepace matchinglib
