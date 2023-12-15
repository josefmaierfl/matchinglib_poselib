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
#include "FileHelper.h"
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

//values for defines: 1 ... Show in window, 2 ... More details
#define DBG_SHOW_MATCHES 0
#define DBG_SHOW_KEYPOINTS 0
#define DBG_SHOW_KP_FILT_MASK 0
#define DBG_SHOW_KEYPOINTS_FILTERED 0
#define DBG_SHOW_ADDITIONAL_KP_GEN 0
#define DBG_SHOW_KEYPOINTS_IMG_SIZE_MAX 640

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
#if DBG_SHOW_KEYPOINTS || DBG_SHOW_KP_FILT_MASK || DBG_SHOW_KEYPOINTS_FILTERED || DBG_SHOW_ADDITIONAL_KP_GEN
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
#endif
#if DBG_SHOW_KP_FILT_MASK
  void visualizeImg(const cv::Mat &img, const std::string &img_baseName, const int img_idx);
#endif
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
                                   std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors,
                                   const bool useCuda);
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

  size_t convert4PixelValsTo32bit(const int &d_x, const int &d_y, const cv::Point &start, const cv::Mat &img);
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

  bool IsDescriptorSupportedByMatcher(const std::string &descriptorType, const std::string &matcherType)
  {
    if(IsBinaryDescriptor(descriptorType))
    {
      return IsBinaryMatcher(matcherType);
    }
    return true;
  }

    /* Image Shadow / Highlight Correction. The same function as it in Photoshop / GIMP
     * adapted version of https://gist.github.com/HViktorTsoi/8e8b0468a9fb07842669aa368382a7df
     * img: input greyscale image
     * shadow_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
     * shadow_tone_percent [0.0 ~ 1.0]: Controls the range of tones (image brightness values, i.e. 0 to shadow_tone_percent * 255) in the shadows that are modified.
     * shadow_radius [>0]: Controls the size of the local neighborhood around each pixel
     * highlight_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
     * highlight_tone_percent [0.0 ~ 1.0]: Controls the range of tones (image brightness values, i.e. 255 - highlight_tone_percent * 255 to 255) in the highlights that are modified.
     * highlight_radius [>0]: Controls the size of the local neighborhood around each pixel
     * histEqual: If true, histogram equalization is performed afterwards
     */
    cv::Mat shadowHighlightCorrection(cv::InputArray img, const float &shadow_amount_percent, const float &shadow_tone_percent, const int &shadow_radius, const float &highlight_amount_percent, const float &highlight_tone_percent, const int &highlight_radius, const bool histEqual)
    {
        const cv::Mat img_in = img.getMat();
        CV_Assert(img_in.type() == CV_8UC1);
        CV_Assert(shadow_amount_percent >= 0 && shadow_amount_percent <= 1.f && shadow_tone_percent >= 0 && shadow_tone_percent <= 1.f);
        CV_Assert(highlight_amount_percent >= 0 && highlight_amount_percent <= 1.f && highlight_tone_percent >= 0 && highlight_tone_percent <= 1.f);
        CV_Assert(shadow_radius > 0 && shadow_radius < std::min(img_in.rows, img_in.cols) / 4);
        CV_Assert(highlight_radius > 0 && highlight_radius < std::min(img_in.rows, img_in.cols) / 4);
        const float shadow_tone = shadow_tone_percent * 255.f;//0...1 -> 0...255
        const float highlight_tone = 255.f - highlight_tone_percent * 255.f; // 0...1 -> 255...0

        const float shadow_gain = 1.f + shadow_amount_percent * 6.f;// 0...1 -> 1...7
        const float highlight_gain = 1.f + highlight_amount_percent * 6.f;//0...1 -> 1...7

        // Convert img to float
        cv::Mat img_flt;
        img_in.convertTo(img_flt, CV_32FC1);
        const cv::Size imgSi = img_in.size();

        // extract shadow
        // darkest regions get highest values, img values > shadow_tone -> 0, range: 0...shadow_tone -> 255...0
        cv::Mat shadow_map = 255.f - (img_flt * 255.f) / shadow_tone;
        // cv::Mat shadow_map = 255.f - img_flt / shadow_tone;
        for (int y = 0; y < imgSi.height; y++)
        {
            for (int x = 0; x < imgSi.width; x++)
            {
                if (img_flt.at<float>(y, x) >= shadow_tone)
                {
                    shadow_map.at<float>(y, x) = 0.f;
                }
            }
        }

        // extract highlight
        // brightest regions get highest values, img values < highlight_tone -> 0, range highlight_tone...255 -> 0...255
        cv::Mat highlight_map = 255.f - ((255.f - img_flt) * 255.f) / (255.f - highlight_tone);
        // cv::Mat highlight_map = 255.f - (255.f - img_flt) / (255.f - highlight_tone);
        for (int y = 0; y < imgSi.height; y++)
        {
            for (int x = 0; x < imgSi.width; x++)
            {
                if (img_flt.at<float>(y, x) <= highlight_tone)
                {
                    highlight_map.at<float>(y, x) = 0.f;
                }
            }
        }

        // Gaussian blur on tone map, for smoother transition
        if (shadow_amount_percent * static_cast<float>(shadow_radius) > 0.f)
        {
            cv::blur(shadow_map, shadow_map, cv::Size(shadow_radius, shadow_radius));
        }
        if (highlight_amount_percent * static_cast<float>(highlight_radius) > 0.f)
        {
            cv::blur(highlight_map, highlight_map, cv::Size(highlight_radius, highlight_radius));
        }

        // Tone LUT
        std::vector<float> t(256);
        std::iota(t.begin(), t.end(), 0);
        std::vector<float> lut_shadow, lut_highlight;
        const float m1 = 1.f / 255.f;
        for (size_t i = 0; i < 256; i++)
        {
            const float im1 = static_cast<float>(i) * m1;
            const float pwr = std::pow(1.f - im1, shadow_gain); //(1 - i / 255)^shadow_gain: 1...0
            float ls = (1.f - pwr) * 255.f;//0...255
            ls = std::max(0.f, std::min(255.f, std::round(ls)));
            lut_shadow.emplace_back(std::move(ls));

            float lh = std::pow(im1, highlight_gain) * 255.f; //(i / 255)^highlight_gain: 0...255
            lh = std::max(0.f, std::min(255.f, std::round(lh)));
            lut_highlight.emplace_back(std::move(lh));
        }

        // adjust tone
        shadow_map = shadow_map * m1;//0...1
        highlight_map = highlight_map * m1;//0...1

        cv::Mat shadow_map_tone1 = cv::Mat::zeros(imgSi, shadow_map.type());
        for (int y = 0; y < imgSi.height; y++)
        {
            for (int x = 0; x < imgSi.width; x++)
            {
                const unsigned char &vi = img_in.at<unsigned char>(y, x);
                const float &ls = lut_shadow.at(vi);
                shadow_map_tone1.at<float>(y, x) = ls * shadow_map.at<float>(y, x); //[0...255] * [0...1] -> [0...255]
            }
        }
        cv::Mat shadow_map_tone2 = 1.f - shadow_map;     // 1...0, 1 for all pixel values > shadow_tone
        shadow_map_tone2 = shadow_map_tone2.mul(img_flt); //[1...0] * [0...255], pixel values > shadow_tone remain untouched
        shadow_map_tone1 += shadow_map_tone2;

        cv::Mat highlight_map_tone1 = cv::Mat::zeros(imgSi, shadow_map.type());
        for (int y = 0; y < imgSi.height; y++)
        {
            for (int x = 0; x < imgSi.width; x++)
            {
                const unsigned char vi = static_cast<unsigned char>(std::max(0.f, std::min(std::round(shadow_map_tone1.at<float>(y, x)), 255.f)));
                const float &lh = lut_highlight.at(vi);
                highlight_map_tone1.at<float>(y, x) = lh * highlight_map.at<float>(y, x);
            }
        }
        cv::Mat highlight_map_tone2 = 1.f - highlight_map;
        highlight_map_tone2 = highlight_map_tone2.mul(shadow_map_tone1);
        shadow_map_tone1 = highlight_map_tone2 + highlight_map_tone1;
        cv::convertScaleAbs(shadow_map_tone1, shadow_map_tone1);
        // cv::imshow("other", shadow_map_tone1);

        cv::Mat histequ;
        if (histEqual)
        {
            cv::equalizeHist(shadow_map_tone1, histequ);
            // cv::imshow("other2", histequ);
        }
        else
        {
            histequ = shadow_map_tone1;
        }

        return histequ;
    }

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
                     const std::string &matcher_type,
                     const std::vector<std::string> &mask_file_names,
                     const std::vector<int> &img_indices,
                     const std::vector<std::pair<int, int>> &match_cams,
                     const bool sort_file_names,
                     const double &img_scale, 
                     const bool equalizeImgs, 
                     const int &nrFeaturesToExtract_,
                     const int &cpuCnt_) : img_scaling(img_scale), 
                                           keypoint_types_(keypoint_types), 
                                           descriptor_type_(descriptor_type), 
                                           matcher_type_(matcher_type),
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
    for(const auto &kpt: keypoint_types_)
    {
      if(!IsKeypointTypeSupported(kpt))
      {
        throw runtime_error("Keypoint type " + kpt + " is not supported");
      }
    }
    if(!IsDescriptorTypeSupported(descriptor_type_))
    {
      throw runtime_error("Descriptor type " + descriptor_type_ + " is not supported");
    }
    if(!IsMatcherSupported(matcher_type_))
    {
      throw runtime_error("Matcher type " + matcher_type_ + " is not supported");
    }
    if(!IsDescriptorSupportedByMatcher(descriptor_type_, matcher_type))
    {
      throw runtime_error("Matcher " + matcher_type_ + " and descriptor " + descriptor_type_ + " are not compatible");
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
          img_mask_names.emplace(i, make_pair(img_file_names.at(i), mask_file_names.at(i)));
        }
        else
        {
          img_mask_names.emplace(i, make_pair(img_file_names.at(i), ""));
        }
      }
    }
    else
    {
      if(img_file_names.size() != indices.size())
      {
        throw runtime_error("Number of indices must be the same as images.");
      }
      std::vector<pair<int, int>> idx_tmp;
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
          img_mask_names.emplace(i.second, make_pair(img_file_names.at(i.first), mask_file_names.at(i.first)));
        }
        else
        {
          img_mask_names.emplace(i.second, make_pair(img_file_names.at(i.first), ""));
        }
      }
    }
    for(const auto &i: indices)
    {
      imageMap.emplace(i, make_pair(cv::Mat(), cv::Mat()));
    }
    if(!match_cams.empty()){
        cam_pair_idx = match_cams;
    }
    
    std::cout << "Loading image data ..." << endl;
    loadImages();
    checkImgSizes();
    imgs_ID = getIDfromImages(imageMap);
  }

  Matching::Matching(const std::vector<cv::Mat> &imgs,
                     const std::vector<std::string> &keypoint_types, 
                     const std::string &descriptor_type,
                     const std::string &matcher_type,
                     const std::vector<cv::Mat> &masks,
                     const std::vector<int> &img_indices,
                     const std::vector<std::pair<int, int>> &match_cams,
                     const double &img_scale, 
                     const bool equalizeImgs, 
                     const int &nrFeaturesToExtract_,
                     const int &cpuCnt_) : img_scaling(img_scale), 
                                           keypoint_types_(keypoint_types), 
                                           descriptor_type_(descriptor_type), 
                                           matcher_type_(matcher_type),
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
    for(const auto &kpt: keypoint_types_)
    {
      if(!IsKeypointTypeSupported(kpt))
      {
        throw runtime_error("Keypoint type " + kpt + " is not supported");
      }
    }
    if(!IsDescriptorTypeSupported(descriptor_type_))
    {
      throw runtime_error("Descriptor type " + descriptor_type_ + " is not supported");
    }
    if(!IsMatcherSupported(matcher_type_))
    {
      throw runtime_error("Matcher type " + matcher_type_ + " is not supported");
    }
    if(!IsDescriptorSupportedByMatcher(descriptor_type_, matcher_type))
    {
      throw runtime_error("Matcher " + matcher_type_ + " and descriptor " + descriptor_type_ + " are not compatible");
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
      std::vector<pair<int, int>> idx_tmp;
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
    if(!match_cams.empty()){
        cam_pair_idx = match_cams;
    }

    std::cout << "Preprocessing image data ..." << endl;
    preprocessImages();
    checkImgSizes();
    imgs_ID = getIDfromImages(imageMap);
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
        scaleEqualizeImg(img1, imageMap[idx].first, img_scaling, equalizeImgs_);
        if(haveMasks)
        {
          cv::Mat img2 = cv::imread(img_mask_names.at(idx).second, cv::IMREAD_GRAYSCALE);
          if (img2.data == NULL)
          {
              throw runtime_error("Unable to read mask image " + img_mask_names.at(idx).second);
          }
          scaleEqualizeImg(img2, imageMap[idx].second, img_scaling, false);
          // if (equalizeImgs_)
          // {
          //     cv::equalizeHist(img2, imageMap[idx].second);
          // }
          // else
          // {
          //     img2.copyTo(imageMap[idx].second);
          // }
        }
    }
  }

  void Matching::preprocessImages()
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
        threads.push_back(std::thread(std::bind(&Matching::preprocessImageThreadFunc, this, startIdx, endIdx)));
    }

    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
    cout << "Preprocessing images took " << timeDiff / 1e3 << " seconds." << endl;
  }

  void Matching::preprocessImageThreadFunc(const int startIdx, const int endIdx)
  {
    for (auto i = startIdx; i < endIdx; ++i)
    {
        const int &idx = indices[i];
        scaleEqualizeImg(imageMap[idx].first, imageMap[idx].first, img_scaling, equalizeImgs_);
        if(haveMasks)
        {
          scaleEqualizeImg(imageMap[idx].second, imageMap[idx].second, img_scaling, false);
          // if (equalizeImgs_)
          // {
          //     cv::equalizeHist(imageMap[idx].second, imageMap[idx].second);
          // }
          // else
          // {
          //     imageMap[idx].second.copyTo(imageMap[idx].second);
          // }
        }
    }
  }

  void Matching::checkImgSizes()
  {
    std::unordered_map<std::pair<int, int>, int, pair_hash, pair_EqualTo> img_sizes;
    for(const auto &img: imageMap){
        cv::Size ims = img.second.first.size();
        std::pair<int, int> imsp(ims.width, ims.height);
        if(img_sizes.find(imsp) == img_sizes.end())
        {
            img_sizes.emplace(imsp, 1);
        }
        else
        {
            img_sizes.at(imsp)++;
        }
    }
    if(img_sizes.size() > 1){
        cout << "WARNING! Input images with different size present. Automatically scaling and cropping images" << endl;
        int w_min = INT_MAX, h_min = INT_MAX, w_max = 0, h_max = 0;
        for(const auto &ims: img_sizes)
        {
            const int &w = ims.first.first;
            const int &h = ims.first.second;
            if(w < w_min){
                w_min = w;
            }
            if(w > w_max){
                w_max = w;
            }
            if(h < h_min){
                h_min = h;
            }
            if(h > h_max){
                h_max = h;
            }
        }
        const double s_w = static_cast<double>(w_min) / static_cast<double>(w_max);
        const double s_h = static_cast<double>(h_min) / static_cast<double>(h_max);
        const bool use_w = (s_w > s_h);
        int w_min2 = INT_MAX, h_min2 = INT_MAX;
        for(auto &img: imageMap)
        {
            cv::Size ims = img.second.first.size();
            double s = 1.0;
            if(use_w){
                if(ims.width > w_min){
                    s = static_cast<double>(w_min) / static_cast<double>(ims.width);
                }else{
                    continue;
                }
                ims.width = w_min;
                ims.height = static_cast<int>(floor(static_cast<double>(ims.height) * s + 0.5));
            }
            else
            {
                if(ims.height > h_min){
                    s = static_cast<double>(h_min) / static_cast<double>(ims.height);
                }else{
                    continue;
                }
                ims.height = h_min;
                ims.width = static_cast<int>(floor(static_cast<double>(ims.width) * s + 0.5));
            }
            cv::resize(img.second.first, img.second.first, ims, 0, 0, cv::INTER_AREA);
            if(!img.second.second.empty())
            {
                cv::resize(img.second.first, img.second.first, ims, 0, 0, cv::INTER_NEAREST);
            }
            if(ims.width < w_min){
                w_min2 = ims.width;
            }
            if(ims.height < h_min){
                h_min2 = ims.height;
            }
        }
        if(w_min2 == INT_MAX){
            w_min2 = w_min;
        }
        if(h_min2 == INT_MAX){
            h_min2 = h_min;
        }
        cv::Rect roi(0, 0, w_min2, h_min2);
        for(auto &img: imageMap)
        {
            cv::Size ims = img.second.first.size();
            if(ims.width > w_min2 || ims.height > h_min2){
                img.second.first = img.second.first(roi);
                if(!img.second.second.empty())
                {
                    img.second.second = img.second.second(roi);
                }
            }
        }
    }
  }

  void scaleEqualizeImg(const cv::Mat &img_in, cv::Mat &img_out, const double &img_scaling, const bool equalizeImg)
  {
    cv::Mat img_tmp = img_in;
    if (!nearZero(img_scaling - 1.0))
    {
        cv::Size imgSi = img_in.size();
        cv::Size newImgSi(static_cast<int>(std::round(img_scaling * static_cast<double>(imgSi.width))), static_cast<int>(std::round(img_scaling * static_cast<double>(imgSi.height))));
        cv::resize(img_in, img_tmp, newImgSi, 0, 0, cv::INTER_AREA);
    }
    if (equalizeImg)
    {
#if USE_CLAHE_FOR_HIST_EQU
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(16,16));
        clahe->apply(img_tmp, img_out);
#else
        cv::equalizeHist(img_tmp, img_out);
#endif
    }
    else
    {
        img_tmp.copyTo(img_out);
    }
  }

  size_t convert4PixelValsTo32bit(const int &d_x, const int &d_y, const cv::Point &start, const cv::Mat &img)
  {
    size_t vals = 0;
    for (int i = 0; i < 4; i++)
    {
      const int x = start.x + i * d_x;
      const int y = start.y + i * d_y;
      const size_t val = static_cast<size_t>(img.at<unsigned char>(y, x));
      vals |= (val << (i * 8));
    }
    return vals;
  }

  std::string getIDfromImages(const std::vector<cv::Mat> &imgs)
  {
    const double borderIgnore = 0.1;
    size_t imgVals[16] = {0};
    for(const auto &img: imgs)
    {
      const cv::Size imgSi = img.size();
      const cv::Point lu(static_cast<int>(std::round(borderIgnore * static_cast<double>(imgSi.width))), static_cast<int>(std::round(borderIgnore * static_cast<double>(imgSi.height))));
      const cv::Point rl(imgSi.width - lu.x, imgSi.height - lu.y);
      const cv::Size newImgSi(rl.x - lu.x, rl.y - lu.y);
      const int d_x = newImgSi.width / 16;
      const int d_y = newImgSi.height / 16;
      const int d_x4 = d_x / 4;
      const int d_y4 = d_y / 4;
      for (int i = 0; i < 16; i++)
      {
        const int i_dx = i * d_x;
        const int x1 = lu.x + i_dx;
        const int x2 = rl.x - i_dx;
        const int y = lu.y + i * d_y;
        size_t val1 = convert4PixelValsTo32bit(d_x4, d_y4, cv::Point(x1, y), img);
        val1 ^= convert4PixelValsTo32bit(-1 * d_x4, d_y4, cv::Point(x2, y), img);
        imgVals[i] ^= val1;
      }
    }

    //Map to 62 different characters [0-9][a-z][A-Z]
    char imgCharVals[17] = {0};
    imgCharVals[16] = '\0';//null-termination to convert it to string later on
    for (int i = 0; i < 16; i++)
    {
      char &val = imgCharVals[i];
      val = static_cast<char>(imgVals[i] % 62);
      if(val < 10){
        val += '0';//maps to [0-9]
      }
      else if(val < 36)
      {
        val += 'A' - static_cast<char>(10);//maps to [A-Z]
      }
      else
      {
        val += 'a' - static_cast<char>(36);//maps to [a-z]
      }
    }
    
    return std::string(imgCharVals);
  }

  std::string getIDfromImages(const std::unordered_map<int, cv::Mat> &images)
  {
    std::vector<cv::Mat> imgs_vec;
    for(const auto &i: images)
    {
      imgs_vec.emplace_back(i.second);
    }
    return getIDfromImages(imgs_vec);
  }
  
  std::string getIDfromImages(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap)
  {
    std::vector<cv::Mat> imgs_vec;
    for(const auto &i: imageMap)
    {
      imgs_vec.emplace_back(i.second.first);
    }
    return getIDfromImages(imgs_vec);
  }    

  bool Matching::compute(const bool affineInvariant, const bool checkNrMatches, const bool addShaddowHighlightSearch, const bool useCuda)
  {
      if (largest_cam_idx < 2)
      {
          std::cerr << "Too less images." << endl;
          return false;
      }
      if (affineInvariant)
      {
          if (keypoint_types_.size() != 1 || keypoint_types_.at(0).compare(descriptor_type_) != 0)
          {
              throw runtime_error("For affine feature detection, only identical keypoint detector (and only 1 type) and descriptor extractor types are supported.");
          }
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
                  visualizeKeypoints(init_keypoints, imageMap, descriptor_type_);
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
      visualizeKeypoints(keypoints_descriptors, imageMap, kp_str);
#endif
      if(addShaddowHighlightSearch){
        findAdditionalKeypointsLessTexture();
      }
      return getMatches(affineInvariant, checkNrMatches);
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
                  visualizeKeypoints(*init_keypoints_tmp_ptr, imageMap_, descriptor_type_);
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
      visualizeKeypoints(keypoints_descriptors_, imageMap_, kp_str);
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

  bool Matching::getMatches(const bool checkNrMatches)
  {
      return getMatches(false, checkNrMatches);
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
#ifdef WITH_AKAZE_CUDA
        bool cuda_ft = IsFeatureCudaTypeSupported(kpt);
#else
        bool cuda_ft = false;
#endif
          // Disable ORB Cuda (use CPU version instead), as keypoints are only extracted approx. on half the image height
          if (useCuda && cuda_ft && kpt.compare("AKAZE") && kpt.compare("ORB"))
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
          std::mt19937 &mt = RandomGenerator::getInstance(std::seed_seq(imgs_ID.begin(), imgs_ID.end())).getTwisterEngineRef();
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

    void getKeypointsAkazeCudaThreadFunc(const int startIdx, 
                                         const int endIdx, 
                                         const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                                         const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                         cv::Ptr<matchinglib::cuda::Feature2DAsync> detector, 
                                         std::shared_ptr<std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>>> keypoints, 
                                         std::exception_ptr &thread_exception, 
                                         std::mt19937 mt)
    {
        try
        {
            cv::cuda::Stream stream;
            size_t nrInvalids = 0;
            // detector.dynamicCast<RR::cuda::AKAZE>()->setDetectResponseTh(detector.dynamicCast<RR::cuda::AKAZE>()->getDetectResponseTh() / 100.f);
            for (auto i = startIdx; i < endIdx; ++i)
            {
                const std::pair<string, int> idx = indices_kp_threads[i];
                const string kp_type = idx.first;
                const int img_idx = idx.second;
                if (kp_type.compare("AKAZE") == 0)
                {
                    std::string info = std::to_string(img_idx);
                    detector.dynamicCast<matchinglib::cuda::AKAZE>()->setImageInfo(info, false);
                }
                if (keypoints->find(kp_type) == keypoints->end())
                {
                    keypoints->emplace(kp_type, std::unordered_map<int, std::vector<cv::KeyPoint>>());
                }
                keypoints->at(kp_type).emplace(img_idx, std::vector<cv::KeyPoint>());
                std::vector<cv::KeyPoint> &keypoints_tmp = keypoints->at(kp_type).at(img_idx);
                // cv::Mat descr;
                detector->detectAndComputeAsync(imageMap_threads->at(img_idx).first, cv::noArray(), keypoints_tmp, cv::noArray(), mt, false, stream);
                if (keypoints_tmp.empty())
                {
                    string msg = "Unable to detect " + kp_type + " keypoints on image " + std::to_string(img_idx) + " using the GPU.";
                    cerr << msg << endl;
                    nrInvalids++;
                    if(nrInvalids > 5){
                        throw runtime_error("Detection of CUDA AKAZE keypoints failed for more than 5 images.");
                    }
                }
            }
            if (nrInvalids)
            {
                throw runtime_error("Detection of CUDA AKAZE keypoints failed for one or more images.");
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
#ifdef WITH_AKAZE_CUDA
      vector<cv::Ptr<matchinglib::cuda::Feature2DAsync>> detector_ptrs_gpu_akaze;
#endif
      if(useCudaOpenCV){
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
          std::mt19937 &mt = RandomGenerator::getInstance(std::seed_seq(imgs_ID.begin(), imgs_ID.end())).getTwisterEngineRef();
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
          std::mt19937 &mt = RandomGenerator::getInstance(std::seed_seq(imgs_ID.begin(), imgs_ID.end())).getTwisterEngineRef();
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
                                 const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                 cv::Ptr<cv::cuda::Feature2DAsync> kp_detect_ptr, 
                                 std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                 std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors)
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
                                      const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                      cv::Ptr<matchinglib::cuda::Feature2DAsync> kp_detect_ptr, 
                                      std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                      std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors, 
                                      std::mt19937 mt)
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

  bool Matching::getFeaturesAffine(const bool useCuda)
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

      int nrKeyPointsMax = getMaxNrFeatures(true);

      auto startTime = std::chrono::steady_clock::now();

      bool ret = getFeaturesAffine(imageMap, indices, indices_kp, keypoints_descriptors, nrKeyPointsMax, useCuda);
      if(!ret){
          return false;
      }

      for(const auto &kpd: keypoints_descriptors){
          init_keypoints.at(descriptor_type_).at(kpd.first) = kpd.second.first;
      }
      keypoints_combined = init_keypoints.at(descriptor_type_);

      kp_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
      cout << "Detecting affine invariant features took " << kp_time_ms / 1e3 << " seconds." << endl;
      return true;
  }

  bool Matching::getFeaturesAffine(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
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
      if (init_keypoints_)
      {
          init_keypoints_->emplace(descriptor_type_, move(kps));
      }

      bool ret = getFeaturesAffine(imageMap_, indices_, indices_kp_tmp, keypoints_descriptors_, nrKeyPointsMax, useCuda);
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
      else if (keypoints_combined_){
          for (const auto &kpd : keypoints_descriptors_)
          {
              keypoints_combined_->emplace(kpd.first, kpd.second.first);
          }
      }
      if (init_keypoints_ && keypoints_combined_){
          *keypoints_combined_ = init_keypoints_->at(descriptor_type_);
      }
      return true;
  }

  bool Matching::getFeaturesAffine(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                                   const std::vector<int> &indices_, 
                                   const std::vector<std::pair<std::string, int>> &indices_kp_, 
                                   std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                                   const int &nrKeyPointsMax, 
                                   const bool useCuda)
    {
      assert(!indices_.empty() && indices_.size() == indices_kp_.size());
#ifdef WITH_AKAZE_CUDA
      const bool useCudaOpenCV = (useCuda && !descriptor_type_.compare("ORB"));
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
      unsigned threadCount = useAkazeCuda ? 1 : std::min(extractionsCount, static_cast<unsigned>(cpuCnt));
      unsigned batchSize = std::ceil(extractionsCount / static_cast<float>(threadCount));
      getThreadBatchSize(extractionsCount, threadCount, batchSize);

      vector<cv::Ptr<cv::Feature2D>> kp_detect_ptrs;
      vector<cv::Ptr<cv::Feature2D>> kp_aff_detect_ptrs;
      vector<std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>>> kps_thread;
      vector<std::shared_ptr<std::unordered_map<int, cv::Mat>>> descr_thread;
      for (unsigned int i = 0; i < threadCount; ++i)
      {
          kps_thread.emplace_back(std::make_shared<std::unordered_map<int, std::vector<cv::KeyPoint>>>());
          descr_thread.emplace_back(std::make_shared<std::unordered_map<int, cv::Mat>>());
      }

      if (useCudaOpenCV || useAkazeCuda)
      {
#ifdef WITH_AKAZE_CUDA
          if (useCudaOpenCV){
              for (unsigned int i = 0; i < threadCount; ++i)
              {
                  kp_detect_ptrs.emplace_back(matchinglib::cuda::ORB::create(nrKeyPointsMax));
                  kp_aff_detect_ptrs.emplace_back(matchinglib::cuda::AffineFeature::create(kp_detect_ptrs.back(), 5, 0, 1.414213538f, 72.f, true, static_cast<int>(threadCount)));
              }
          }
          else
          {
              cv::Size imgSi = imageMap_.begin()->second.first.size();
              kp_detect_ptrs.emplace_back(matchinglib::cuda::AKAZE::create(imgSi.width, imgSi.height));
              kp_detect_ptrs.back().dynamicCast<matchinglib::cuda::AKAZE>()->setMaxNrKeypoints(nrKeyPointsMax);
              kp_aff_detect_ptrs.emplace_back(matchinglib::cuda::AffineFeature::create(kp_detect_ptrs.back(), 5, 0, 1.414213538f, 72.f, true));
          }
#else
          throw runtime_error("Was compiled without CUDA.");
#endif
      }else{
          for (unsigned int i = 0; i < threadCount; ++i)
          {
              kp_detect_ptrs.emplace_back(createExtractor(descriptor_type_, descriptor_type_, nrKeyPointsMax));
              kp_aff_detect_ptrs.emplace_back(cv::AffineFeature::create(kp_detect_ptrs.back()));
          }
      }

      for (unsigned int i = 0; i < threadCount; ++i)
      {
          const int startIdx = i * batchSize;
          const int endIdx = std::min((i + 1u) * batchSize, extractionsCount);
          threads.push_back(std::thread(std::bind(&getFeaturesAffineThreadFunc, startIdx, endIdx, indices_kp_, &imageMap_, kp_aff_detect_ptrs.at(i), kps_thread.at(i), descr_thread.at(i), (useCudaOpenCV || useAkazeCuda))));
      }

      for (auto &t : threads)
      {
          if (t.joinable())
          {
              t.join();
          }
      }

      for (unsigned int i = 0; i < threadCount; ++i){
          for (const auto &kp_ci : *kps_thread.at(i)){
              if(kp_ci.second.empty()){
                  return false;
              }
              keypoints_descriptors_.at(kp_ci.first).first = kp_ci.second;
              keypoints_descriptors_.at(kp_ci.first).second = descr_thread.at(i)->at(kp_ci.first).clone();
          }
      }

      return true;
  }

  void getFeaturesAffineThreadFunc(const int startIdx, 
                                   const int endIdx, 
                                   const std::vector<std::pair<std::string, int>> indices_kp_threads, 
                                   const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> *imageMap_threads, 
                                   cv::Ptr<cv::Feature2D> kp_detect_ptr, 
                                   std::shared_ptr<std::unordered_map<int, std::vector<cv::KeyPoint>>> keypoints, 
                                   std::shared_ptr<std::unordered_map<int, cv::Mat>> descriptors,
                                   const bool useCuda)
  {
      for (auto i = startIdx; i < endIdx; ++i)
      {
          const std::pair<string, int> idx = indices_kp_threads.at(i);
          const int img_idx = idx.second;
#ifdef WITH_AKAZE_CUDA
          if(useCuda){
            std::string info = std::to_string(img_idx);
            if (std::get<0>(idx).compare("AKAZE") == 0)
            {
                kp_detect_ptr.dynamicCast<matchinglib::cuda::AffineFeature>()->getBackendPtr().dynamicCast<matchinglib::cuda::AKAZE>()->setImageInfo(info, false);
            }
            kp_detect_ptr.dynamicCast<matchinglib::cuda::AffineFeature>()->setImgInfoStr(info);
          }
#endif
          descriptors->emplace(img_idx, cv::Mat());
          keypoints->emplace(img_idx, std::vector<cv::KeyPoint>());
          cv::Mat &descr_tmp = descriptors->at(img_idx);
          std::vector<cv::KeyPoint> &kps_tmp = keypoints->at(img_idx);
          kp_detect_ptr->detectAndCompute(imageMap_threads->at(img_idx).first, cv::noArray(), kps_tmp, descr_tmp);
          unordered_set<int> del_idx;
          cv::Size imgSi = imageMap_threads->at(img_idx).first.size();
          imgSi.width -= 1;
          imgSi.height -= 1;
          for (int j = 0; j < descr_tmp.rows; j++)
          {
              const cv::Point2f &pt = kps_tmp.at(j).pt;
              if (pt.x < 0 || pt.y < 0 || pt.x > static_cast<float>(imgSi.width) || pt.y > static_cast<float>(imgSi.height))
              {
                  del_idx.emplace(j);
              }
          }
          if(!del_idx.empty()){
              std::vector<cv::KeyPoint> keypoints_fd1;
              cv::Mat descr;
              for (int j = 0; j < descr_tmp.rows; j++)
              {
                  if (del_idx.find(j) != del_idx.end()){
                      continue;
                  }
                  keypoints_fd1.emplace_back(kps_tmp.at(j));
                  descr.push_back(descr_tmp.row(j));
              }
              kps_tmp = keypoints_fd1;
              descr.copyTo(descr_tmp);
          }
      }
  }

#define MATCHING_INTERNAL_THREADS 0
  bool Matching::getMatches(const bool affineInvariant, const bool checkNrMatches)
  {
      // Get matching permutations
      if(cam_pair_idx.empty()){
        for (int c1 = 0; c1 < largest_cam_idx - 1; c1++)
        {
            for (int c2 = c1 + 1; c2 < largest_cam_idx; c2++)
            {
                if (c1 == 0 && c2 == (largest_cam_idx - 1))
                {
                    continue;
                }
                cam_pair_idx.emplace_back(make_pair(c1, c2));
            }
        }
        cam_pair_idx.emplace_back(make_pair(largest_cam_idx - 1, 0));
      }

      std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> kp_descr_mask;
      if(haveMasks)
      {
        applyMaskToFeatures(false, kp_descr_mask);
      }
      else
      {
        kp_descr_mask = keypoints_descriptors;
      }
      keypoints_descriptors = move(kp_descr_mask);

      auto startTime = std::chrono::steady_clock::now();
      std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> keypoints_descriptors_sv;
      bool data_without_filtering = false;
      if (affineInvariant)
      {
          data_without_filtering = true;
          filterResponseAffine();
          //Copy data for backup
          for (auto &mdata : keypoints_descriptors)
          {
              keypoints_descriptors_sv.emplace(mdata.first, make_pair(mdata.second.first, mdata.second.second.clone()));
          }
          filterResponseAreaBasedAffine(2 * nrFeaturesToExtract);
      }
      auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
      cout << "First feature filtering step took " << timeDiff / 1e3 << " seconds." << endl;

      // KEYPOINT_TYPES, DESCRIPTOR_TYPES: first cam camera index, first cam image index, second cam camera index, second cam image index: matches
      std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> matches;
      std::mt19937 &mt = RandomGenerator::getInstance(std::seed_seq(imgs_ID.begin(), imgs_ID.end())).getTwisterEngineRef();
      do
      {
          for (auto &mi : cam_pair_idx)
          {
              matches.emplace(mi, std::vector<cv::DMatch>());
          }

          startTime = std::chrono::steady_clock::now();
#if !MATCHING_INTERNAL_THREADS
          std::vector<std::thread> threads;
#endif
          const unsigned matchImgsCount = cam_pair_idx.size();
          unsigned threadCount = std::min(matchImgsCount, static_cast<unsigned>(cpuCnt));
          unsigned batchSize = std::ceil(matchImgsCount / static_cast<float>(threadCount));
          getThreadBatchSize(matchImgsCount, threadCount, batchSize);
          std::vector<std::exception_ptr> thread_exceptions(threadCount, nullptr);

//To get rid of repeatability issue
#if !MATCHING_INTERNAL_THREADS
          Matching::getMatchesThreadFunc(0, 1, &keypoints_descriptors, &matches, thread_exceptions.at(0), mt);
          try
          {
              if (thread_exceptions.at(0))
              {
                  std::rethrow_exception(thread_exceptions.at(0));
              }
          }
          catch (const std::exception &e)
          {
              cerr << "Exception during matching: " << e.what() << endl;
              return false;
          }
#endif

          for (unsigned int i = 0; i < threadCount; ++i)
          {
              int startIdx = i * batchSize;
              const int endIdx = std::min((i + 1u) * batchSize, matchImgsCount);
#if MATCHING_INTERNAL_THREADS
              Matching::getMatchesThreadFunc(startIdx, endIdx, &keypoints_descriptors, &matches, thread_exceptions.at(i), mt);
#else
              if(i == 0){
                  startIdx++;
              }
              threads.push_back(std::thread(std::bind(&Matching::getMatchesThreadFunc, this, startIdx, endIdx, &keypoints_descriptors, &matches, thread_exceptions.at(i), mt)));
#endif
          }

#if !MATCHING_INTERNAL_THREADS
          for (auto &t : threads)
          {
              if (t.joinable())
              {
                  t.join();
              }
          }
#endif

          try
          {
              for (auto &e : thread_exceptions)
              {
                  if (e)
                  {
                      std::rethrow_exception(e);
                  }
              }
          }
          catch (const std::exception &e)
          {
              cerr << "Exception during matching: " << e.what() << endl;
              return false;
          }

          match_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
          cout << "Calculating matches took " << match_time_ms / 1e3 << " seconds." << endl;

          bool enough_matches = checkNrMatches ? checkNrMatchesFunc(matches):true;

          if (!enough_matches && !data_without_filtering)
          {
              cout << "Too less matches found." << endl;
              return false;
          }
          else if (!enough_matches && data_without_filtering)
          {
              cout << "Too less matches found. Trying without pre-filtered keypoints ..." << endl;
              data_without_filtering = false;
              //Copy unfiltered data back
              keypoints_descriptors = move(keypoints_descriptors_sv);
              matches.clear();
          }else{
              break;
          }
      } while (true);

      startTime = std::chrono::steady_clock::now();
      if (affineInvariant)
      {
          filterKeypointsClassIDAffine(keypoints_descriptors, matches);
      }

      if(!filterMatches(keypoints_descriptors, matches))
      {
          cout << "Too less matches remaining after filtering." << endl;
          return false;
      }

      if(affineInvariant){
          filterAreaBasedAffine();
          // filterResponseAffine();
          filterResponseAreaBasedAffine();
      }

#if DBG_SHOW_KEYPOINTS_FILTERED
      string kp_str = "filtered_";
      for (const auto &kpt : keypoint_types_)
      {
          kp_str += kpt + "_";
      }
      kp_str += descriptor_type_;
      visualizeKeypoints(keypoints_descriptors, imageMap, kp_str);
#endif

      timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
      cout << "Concatonation/filtering of all matches took " << timeDiff / 1e3 << " seconds." << endl;
      return true;
  }

  void Matching::getMatchesThreadFunc(const int startIdx, 
                                      const int endIdx, 
                                      const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> *kp_descr_in, 
                                      std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> *matches_out, 
                                      std::exception_ptr &thread_exception, 
                                      std::mt19937 mt)
  {
      try
      {
          for (auto i = startIdx; i < endIdx; ++i)
          {
              const std::pair<int, int> &idx = cam_pair_idx.at(i);
              const int img1_idx = idx.first;
              if (imageMap.find(img1_idx) == imageMap.end())
              {
                  continue;
              }
              const int img2_idx = idx.second;
              if (imageMap.find(img2_idx) == imageMap.end())
              {
                  continue;
              }
              const std::vector<cv::KeyPoint> &kps1 = kp_descr_in->at(img1_idx).first;
              const std::vector<cv::KeyPoint> &kps2 = kp_descr_in->at(img2_idx).first;
              const cv::Mat descr1 = kp_descr_in->at(img1_idx).second;
              const cv::Mat descr2 = kp_descr_in->at(img2_idx).second;

              cv::Size imgSi = imageMap.at(img1_idx).first.size();
              std::vector<cv::DMatch> &matches_out_ref = matches_out->at(idx);
#if MATCHING_INTERNAL_THREADS
              unsigned threadCount = static_cast<unsigned>(cpuCnt);
              // To get the same results from run to run using the same data, set parameter "indexThreadQty=1",
              // For speedup set it to threadCount
              string index_pars = "M=50,efConstruction=10,delaunay_type=1,indexThreadQty=1"; // + std::to_string(threadCount);
              string query_pars = "efSearch=10";
              // string index_pars = "indexThreadQty=" + std::to_string(threadCount);
              int err = matchinglib::getMatches(kps1, kps2, descr1, descr2, imgSi, matches_out_ref, mt, matcher_type_, false, true, descriptor_type_, index_pars, query_pars, threadCount); // std::max(std::thread::hardware_concurrency(), 1u))
#else
              string index_pars = "M=50,efConstruction=10,delaunay_type=1,indexThreadQty=1"; // + std::to_string(threadCount);
              string query_pars = "efSearch=10";
              int err = matchinglib::getMatches(kps1, kps2, descr1, descr2, imgSi, matches_out_ref, mt, matcher_type_, false, true, descriptor_type_, index_pars, query_pars, 1u);
#endif
              if (err)
              {
                  string msg = "Unable to calculate matches with " + descriptor_type_ + " descriptor for image pair (" + std::to_string(img1_idx) + "-" + std::to_string(img2_idx) + ")";
                  throw runtime_error(msg);
              }
#if DBG_SHOW_MATCHES
              drawMatchesImgPair(imageMap.at(img1_idx).first, imageMap.at(img2_idx).first, kps1, kps2, matches_out_ref, img1_idx, img2_idx, 200);
#endif
          }
      }
      catch (...)
      {
          thread_exception = std::current_exception();
          return;
      }
  }

  void Matching::applyMaskToFeatures(bool invertMask, std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &kp_descr_out)
  {
      kp_descr_out.clear();
      for (auto &idx : indices)
      {
          cv::Mat mask;
          if (invertMask)
          {
              cv::bitwise_not(imageMap[idx].second, mask);
          }
          else
          {
              mask = imageMap[idx].second;
          }
          kp_descr_out.emplace(idx, make_pair(std::vector<cv::KeyPoint>(), cv::Mat()));
          cv::Mat &descr_out = kp_descr_out[idx].second;
          std::vector<cv::KeyPoint> &kp_out = kp_descr_out[idx].first;
          std::vector<cv::KeyPoint> &kp_ref = keypoints_descriptors[idx].first;
          cv::Mat descr = keypoints_descriptors[idx].second;
          for (size_t i = 0; i < kp_ref.size(); ++i)
          {
              int row = static_cast<int>(std::round(kp_ref[i].pt.y));
              int col = static_cast<int>(std::round(kp_ref[i].pt.x));
              if (mask.at<unsigned char>(row, col) > 0)
              {
                  kp_out.push_back(kp_ref[i]);
                  descr_out.push_back(descr.row(static_cast<int>(i)));
              }
          }
      }
  }

  bool Matching::filterMatches(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &kp_descr_in, 
                               std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> &matches_in)
  {
      for (auto &cpi : cam_pair_idx)
      {
          MatchSearch m_filtered;
          std::vector<cv::DMatch> &mi = matches_in.at(cpi);
          if (mi.empty())
          {
              continue;
          }
          const std::pair<std::vector<cv::KeyPoint>, cv::Mat> &feature_data1 = kp_descr_in.at(cpi.first);
          const std::pair<std::vector<cv::KeyPoint>, cv::Mat> &feature_data2 = kp_descr_in.at(cpi.second);
          const std::vector<cv::KeyPoint> &kps1 = feature_data1.first;
          const std::vector<cv::KeyPoint> &kps2 = feature_data2.first;
          const cv::Mat descr1 = feature_data1.second;
          const cv::Mat descr2 = feature_data2.second;
          for (const auto &m : mi)
          {
              m_filtered.addMatch(kps1.at(m.queryIdx), kps2.at(m.trainIdx), m, descr1.row(m.queryIdx), descr2.row(m.trainIdx));
          }
          MatchData mdata;
          m_filtered.composeAll(mdata.matches, mdata.kps1, mdata.kps2, mdata.descr1, mdata.descr2);
          if (mdata.matches.size() < 50)
          {
              cout << "Only " << mdata.matches.size() << " matches remaining for cam pair (" << cpi.first << "-" << cpi.second << ")." << endl;
              return false;
          }
          matches_filt.emplace(cpi, move(mdata));
      }

      return true;
  }

  void Matching::filterAreaBasedAffine()
  {
      const cv::Size imgSi = imageMap.begin()->second.first.size();
      std::vector<cv::Mat> mask_kp(largest_cam_idx);
      for (int i = 0; i < largest_cam_idx; i++)
      {
          mask_kp[i] = cv::Mat::zeros(imgSi, CV_8UC1);
      }
      for (const auto &mdata : matches_filt){
          const std::vector<cv::KeyPoint> &kp1 = mdata.second.kps1;
          const std::vector<cv::KeyPoint> &kp2 = mdata.second.kps2;
          cv::Mat &mask1 = mask_kp[mdata.first.first];
          cv::Mat &mask2 = mask_kp[mdata.first.second];
          for (const auto &m : mdata.second.matches){
              const cv::Point2f &pt1f = kp1.at(m.queryIdx).pt;
              cv::Point2i pt1(static_cast<int>(round(pt1f.x)), static_cast<int>(round(pt1f.y)));
              mask1.at<unsigned char>(pt1) = UCHAR_MAX;

              const cv::Point2f &pt2f = kp2.at(m.trainIdx).pt;
              cv::Point2i pt2(static_cast<int>(round(pt2f.x)), static_cast<int>(round(pt2f.y)));
              mask2.at<unsigned char>(pt2) = UCHAR_MAX;
          }
      }
      int kernel_si = max(min(imgSi.height, imgSi.width) / 100, 5);
      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kernel_si + 1, 2 * kernel_si + 1), cv::Point(kernel_si, kernel_si));
      for (int i = 0; i < largest_cam_idx; i++)
      {
          cv::dilate(mask_kp[i], mask_kp[i], kernel);
#if DBG_SHOW_KP_FILT_MASK
          string kp_str = "kp_filt_mask_";
          string kp_type_str;
          for (const auto &kpt : keypoint_types_)
          {
              kp_type_str += kpt + "_";
          }
          kp_type_str += descriptor_type_;
          kp_str += kp_type_str;
          visualizeImg(mask_kp[i], kp_str, i);
          std::vector<cv::KeyPoint> kp_all_virt_imgs;
          for (const auto &mdata : keypoints_descriptors)
          {
              if(mdata.first == i){
                  kp_all_virt_imgs.insert(kp_all_virt_imgs.end(), mdata.second.first.begin(), mdata.second.first.end());
              }
          }
          visualizeKeypoints(imageMap.at(i).first, kp_all_virt_imgs, std::to_string(i), kp_type_str);
#endif
      }

      for (auto &mdata : keypoints_descriptors)
      {
          std::vector<cv::KeyPoint> &kps = mdata.second.first;
          cv::Mat &descr = mdata.second.second;
          const cv::Mat mask = mask_kp[mdata.first];

          std::vector<cv::KeyPoint> kpsNew;
          cv::Mat descrNew;
          for (int i = 0; i < static_cast<int>(kps.size()); i++)
          {
              const auto &kp = kps.at(i);
              cv::Point2i pt(static_cast<int>(round(kp.pt.x)), static_cast<int>(round(kp.pt.y)));
              if(mask.at<unsigned char>(pt)){
                  kpsNew.emplace_back(kp);
                  descrNew.push_back(descr.row(i));
              }
          }
          kps = move(kpsNew);
          descr = descrNew;
      }
  }

  void Matching::filterResponseAffine()
  {
      filterResponseAffine(keypoints_descriptors);
  }

  void Matching::filterResponseAffine(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features_map)
  {
      std::vector<std::thread> threads;
      const unsigned filterCount = features_map.size();
      unsigned threadCount = std::min(filterCount, static_cast<unsigned>(cpuCnt));
      unsigned batchSize = std::ceil(filterCount / static_cast<float>(threadCount));
      getThreadBatchSize(filterCount, threadCount, batchSize);
      std::shared_ptr<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>> filtered_features = make_shared<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>>(std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>());
      for (unsigned int i = 0; i < filterCount; ++i)
      {
          filtered_features->emplace_back(make_shared<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>(make_pair(std::vector<cv::KeyPoint>(), cv::Mat())));
      }

      for (unsigned int i = 0; i < threadCount; ++i)
      {
          const int startIdx = i * batchSize;
          const int endIdx = std::min((i + 1u) * batchSize, filterCount);
          threads.push_back(std::thread(std::bind(&filterResponseAffineThreadFunc, startIdx, endIdx, &features_map, filtered_features)));
      }

      for (auto &t : threads)
      {
          if (t.joinable())
          {
              t.join();
          }
      }

      std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>::iterator it = features_map.begin();
      for (unsigned i = 0; i < filterCount; i++)
      {
          it->second.first = std::move(filtered_features->at(i)->first);
          filtered_features->at(i)->second.copyTo(it->second.second);
          it++;
      }

      // for (auto &mdata : features_map)
      // {
      //     std::vector<cv::KeyPoint> &kps = mdata.second.first;
      //     cv::Mat &descr = mdata.second.second;

      //     KeypointSearch ks;
      //     for (int i = 0; i < static_cast<int>(kps.size()); i++)
      //     {
      //         ks.add(kps.at(i), descr.row(i));
      //     }
      //     ks.composeAll(kps, descr);
      // }
  }

  void filterResponseAffineThreadFunc(const int startIdx, 
                                      const int endIdx, 
                                      const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> *keypoints_descriptors, 
                                      std::shared_ptr<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>> filtered_features)
  {
      std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>::const_iterator it = keypoints_descriptors->begin();
      if (startIdx)
      {
          std::advance(it, startIdx);
      }
      for (int i = startIdx; i < endIdx; i++)
      {
          const std::vector<cv::KeyPoint> &kps = it->second.first;
          const cv::Mat &descr = it->second.second;
          if (descr.rows > 100)
          {
              KeypointSearch ks;
              for (int i = 0; i < static_cast<int>(kps.size()); i++)
              {
                  ks.add(kps.at(i), descr.row(i));
              }
              ks.composeAll(filtered_features->at(i)->first, filtered_features->at(i)->second);
          }
          else if (!descr.empty())
          {
              filtered_features->at(i)->first = kps;
              descr.copyTo(filtered_features->at(i)->second);
          }
          it++;
      }
  }

  void Matching::filterResponseAreaBasedAffine(const int &limitNrFeatures)
  {
      std::unordered_map<int, int> individual_limits;
      for (const auto &kpd : keypoints_descriptors){
          individual_limits.emplace(kpd.first, limitNrFeatures);
      }
      filterResponseAreaBasedAffineImpl(keypoints_descriptors, individual_limits);
  }

  void Matching::filterResponseAreaBasedAffineImpl(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features_map, 
                                                   const std::unordered_map<int, int> &individual_limits)
  {
      std::vector<std::thread> threads;
      const unsigned filterCount = features_map.size();
      unsigned threadCount = std::min(filterCount, static_cast<unsigned>(cpuCnt));
      unsigned batchSize = std::ceil(filterCount / static_cast<float>(threadCount));
      getThreadBatchSize(filterCount, threadCount, batchSize);
      std::shared_ptr<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>> filtered_features = make_shared<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>>(std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>());
      for (unsigned int i = 0; i < filterCount; ++i)
      {
          filtered_features->emplace_back(make_shared<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>(make_pair(std::vector<cv::KeyPoint>(), cv::Mat())));
      }

      const cv::Size imgSize = imageMap.begin()->second.first.size();
      for (unsigned int i = 0; i < threadCount; ++i)
      {
          const int startIdx = i * batchSize;
          const int endIdx = std::min((i + 1u) * batchSize, filterCount);
          threads.push_back(std::thread(std::bind(static_cast<void (&)(const int, 
                                                                       const int, 
                                                                       const std::unordered_map<int, int> &, 
                                                                       const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> *, 
                                                                       const cv::Size, std::shared_ptr<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>>)>(filterResponseAreaBasedAffineThreadFunc), 
                                                  startIdx, endIdx, individual_limits, &features_map, imgSize, filtered_features)));
      }

      for (auto &t : threads)
      {
          if (t.joinable())
          {
              t.join();
          }
      }

      std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>::iterator it = features_map.begin();
      for (unsigned i = 0; i < filterCount; i++)
      {
          it->second.first = std::move(filtered_features->at(i)->first);
          filtered_features->at(i)->second.copyTo(it->second.second);
          it++;
      }

      // for (auto &mdata : features_map)
      // {
      //     std::vector<cv::KeyPoint> kps = mdata.second.first;
      //     PointSearchSimpleQ pss;
      //     for (int i = 0; i < static_cast<int>(kps.size()); i++)
      //     {
      //         pss.add(kps.at(i).pt, i);
      //     }
      //     cv::Mat &descr = mdata.second.second;
      //     matchinglib::responseFilterGridBased(kps, imageMap.at(mdata.first).first.size(), limitNrFeatures);

      //     cv::Mat descr2;
      //     for (const auto &kp : kps)
      //     {
      //         int idx = pss.getIdx(kp.pt);
      //         descr2.push_back(descr.row(idx));
      //     }
      //     descr2.copyTo(descr);
      //     mdata.second.first = move(kps);
      // }
  }

  void filterResponseAreaBasedAffineThreadFunc(const int startIdx, 
                                               const int endIdx, 
                                               const std::unordered_map<int, int> &individual_limits, 
                                               const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> *keypoints_descriptors, 
                                               const cv::Size imgSize, 
                                               std::shared_ptr<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>> filtered_features)
  {
      std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>::const_iterator it = keypoints_descriptors->begin();
      if (startIdx)
      {
          std::advance(it, startIdx);
      }
      for (int i = startIdx; i < endIdx; i++)
      {
          std::vector<cv::KeyPoint> kps = it->second.first;
          const cv::Mat &descr = it->second.second;
          if (descr.rows > 100)
          {
              PointSearchSimpleQ pss;
              for (int j = 0; j < static_cast<int>(kps.size()); j++)
              {
                  pss.add(kps.at(j).pt, j);
              }
              const int limitNrFeatures = individual_limits.at(it->first);
              matchinglib::responseFilterGridBased(kps, imgSize, limitNrFeatures);

              cv::Mat &descr2 = filtered_features->at(i)->second;
              for (const auto &kp : kps)
              {
                  int idx = pss.getIdx(kp.pt);
                  descr2.push_back(descr.row(idx));
              }
              filtered_features->at(i)->first = move(kps);
          }
          else if (!descr.empty())
          {
              filtered_features->at(i)->first = kps;
              descr.copyTo(filtered_features->at(i)->second);
          }
          it++;
      }
  }

  void filterResponseAreaBasedAffineThreadFunc(const int startIdx, 
                                               const int endIdx, 
                                               const int limitNrFeatures, 
                                               const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> *keypoints_descriptors, 
                                               const cv::Size imgSize, 
                                               std::shared_ptr<std::vector<std::shared_ptr<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>> filtered_features)
  {
      std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>::const_iterator it = keypoints_descriptors->begin();
      if(startIdx){
          std::advance(it, startIdx);
      }
      std::unordered_map<int, int> individual_limits;
      for (int i = startIdx; i < endIdx; i++)
      {
          individual_limits.emplace(it->first, limitNrFeatures);
          it++;
      }
      filterResponseAreaBasedAffineThreadFunc(startIdx, endIdx, individual_limits, keypoints_descriptors, imgSize, filtered_features);
  }

  bool Matching::checkNrMatchesFunc(const std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> &matches_in)
  {
      size_t nr_pairs = cam_pair_idx.size();
      size_t bad_cnt = 0;
      for (auto &cpi : cam_pair_idx)
      {
          const std::vector<cv::DMatch> &mi = matches_in.at(cpi);
          if (mi.size() < 50)
          {
              bad_cnt++;
          }
      }
      float bad_ratio = static_cast<float>(bad_cnt) / static_cast<float>(nr_pairs);

      return bad_ratio < 0.33f;
  }

  void Matching::filterKeypointsClassIDAffine(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &kp_descr_in, 
                                              std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> &matches_in)
  {
      std::unordered_map<int, std::set<int>> kp_descr_valid_indices;
      AffineMatchesFilterData data(kp_descr_valid_indices);

      std::vector<std::thread> threads;
      const unsigned imgsCount = cam_pair_idx.size();
      unsigned threadCount = std::min(imgsCount, static_cast<unsigned>(cpuCnt));
      unsigned batchSize = std::ceil(imgsCount / static_cast<float>(threadCount));
      getThreadBatchSize(imgsCount, threadCount, batchSize);

      for (unsigned int i = 0; i < threadCount; ++i)
      {
          const int startIdx = i * batchSize;
          const int endIdx = std::min((i + 1u) * batchSize, imgsCount);
          threads.push_back(std::thread(std::bind(&Matching::filterKeypointsClassIDAffineThreadFunc, this, startIdx, endIdx, std::ref(kp_descr_in), std::ref(matches_in), std::ref(data))));
      }

      for (auto &t : threads)
      {
          if (t.joinable())
          {
              t.join();
          }
      }

      std::unordered_map<int, unordered_map<int, int>> kp_descr_valid_idx_map;
      for (const auto &vi : kp_descr_valid_indices)
      {
          std::pair<std::vector<cv::KeyPoint>, cv::Mat> &feature_data = kp_descr_in.at(vi.first);
          std::vector<cv::KeyPoint> &kps = feature_data.first;
          cv::Mat &descr = feature_data.second;

          std::vector<cv::KeyPoint> kpsNew;
          cv::Mat descrNew;
          int idx = 0;
          unordered_map<int, int> idxOldNew;
          for (const auto &vi1 : vi.second)
          {
              kpsNew.emplace_back(kps.at(vi1));
              descrNew.push_back(descr.row(vi1));
              idxOldNew.emplace(vi1, idx++);
          }
          kps = move(kpsNew);
          descr = descrNew;
          kp_descr_valid_idx_map.emplace(vi.first, move(idxOldNew));
      }

      for (const auto &cpi : cam_pair_idx)
      {
          std::vector<cv::DMatch> &mi = matches_in.at(cpi);
          if (mi.empty())
          {
              continue;
          }
          const unordered_map<int, int> &idxOldNew1 = kp_descr_valid_idx_map.at(cpi.first);
          const unordered_map<int, int> &idxOldNew2 = kp_descr_valid_idx_map.at(cpi.second);
          std::vector<cv::DMatch> matchesNew;
          for (const auto &m : mi)
          {
              if (idxOldNew1.find(m.queryIdx) != idxOldNew1.end() && idxOldNew2.find(m.trainIdx) != idxOldNew2.end())
              {
                  cv::DMatch m1 = m;
                  m1.queryIdx = idxOldNew1.at(m.queryIdx);
                  m1.trainIdx = idxOldNew2.at(m.trainIdx);
                  matchesNew.emplace_back(move(m1));
              }
          }
          mi = move(matchesNew);
      }
  }

  void Matching::filterKeypointsClassIDAffineThreadFunc(const int startIdx, 
                                                        const int endIdx, 
                                                        const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &kp_descr_in, 
                                                        const std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> &matches_in, 
                                                        AffineMatchesFilterData &data)
  {
      for (int icp = startIdx; icp < endIdx; icp++)
      {
          const std::pair<int, int> &cpi = cam_pair_idx.at(icp);
          const std::vector<cv::DMatch> &mi = matches_in.at(cpi);
          if (mi.empty())
          {
              continue;
          }
          const int img1_idx = cpi.first;
          const int img2_idx = cpi.second;
          data.find_emplace(img1_idx);
          data.find_emplace(img2_idx);
          std::vector<cv::DMatch> mi1 = mi;
          const std::pair<std::vector<cv::KeyPoint>, cv::Mat> &feature_data1 = kp_descr_in.at(img1_idx);
          const std::pair<std::vector<cv::KeyPoint>, cv::Mat> &feature_data2 = kp_descr_in.at(img2_idx);
          const std::vector<cv::KeyPoint> &kps1 = feature_data1.first;
          const std::vector<cv::KeyPoint> &kps2 = feature_data2.first;
          const cv::Mat descr1 = feature_data1.second;
          const cv::Mat descr2 = feature_data2.second;
          const cv::Mat &img = imageMap.at(img1_idx).first;
          matchinglib::filterMatchesSOF(kps1, kps2, img.size(), mi1);
          unordered_map<int, int> classIds[2];
          for (const auto &m : mi1)
          {
              const int classID1 = kps1.at(m.queryIdx).class_id;
              const int classID2 = kps2.at(m.trainIdx).class_id;
              if (classIds[0].find(classID1) == classIds[0].end())
              {
                  classIds[0].emplace(classID1, 1);
              }
              else
              {
                  classIds[0].at(classID1)++;
              }
              if (classIds[1].find(classID2) == classIds[1].end())
              {
                  classIds[1].emplace(classID2, 1);
              }
              else
              {
                  classIds[1].at(classID2)++;
              }
          }
          vector<int> classIDs_del[2];
          for (int j = 0; j < 2; j++)
          {
              for (const auto &cid : classIds[j])
              {
                  if (cid.second < 5)
                  {
                      classIDs_del[j].emplace_back(j);
                  }
              }
              for (const auto &di : classIDs_del[j])
              {
                  classIds[j].erase(di);
              }
          }
          FeatureKDTree kdtree[2];
          for (const auto &m : mi)
          {
              kdtree[0].add(kps1.at(m.queryIdx), descr1.row(m.queryIdx));
              kdtree[1].add(kps2.at(m.trainIdx), descr2.row(m.trainIdx));
          }
          kdtree[0].buildIndex();
          kdtree[1].buildIndex();
          for (int j = 0; j < descr1.rows; j++)
          {
              const cv::KeyPoint &kp = kps1.at(j);
              if (classIds[0].find(kp.class_id) == classIds[0].end())
              {
                  continue;
              }
              std::vector<cv::KeyPoint> kp2s;
              if (kdtree[0].getKeypointDescriptorMatches(kp.pt, kp2s, 900.f))
              {
                  for (const auto &kp2 : kp2s)
                  {
                      if (kp2.class_id == kp.class_id)
                      {
                          data.emplaceIdx(img1_idx, j);
                          break;
                      }
                  }
              }
              else
              {
                  data.emplaceIdx(img1_idx, j);
              }
          }
          for (int j = 0; j < descr2.rows; j++)
          {
              const cv::KeyPoint &kp = kps2.at(j);
              if (classIds[1].find(kp.class_id) == classIds[1].end())
              {
                  continue;
              }
              std::vector<cv::KeyPoint> kp2s;
              if (kdtree[1].getKeypointDescriptorMatches(kp.pt, kp2s, 900.f))
              {
                  for (const auto &kp2 : kp2s)
                  {
                      if (kp2.class_id == kp.class_id)
                      {
                          data.emplaceIdx(img2_idx, j);
                          break;
                      }
                  }
              }
              else
              {
                  data.emplaceIdx(img2_idx, j);
              }
          }
      }
  }

    bool Matching::findAdditionalKeypointsLessTexture()
    {
        auto startTime = std::chrono::steady_clock::now();
        std::vector<int> camImgIdxsSearchAddKp;
        std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> imageMap_boosted;
        unordered_map<int, cv::Rect> imgROIs;

        unordered_map<int, cv::Mat> descr_add;
        unordered_map<int, std::vector<cv::KeyPoint>> keypoints_add;

        for (int c = 0; c < largest_cam_idx; c++)
        {
            if (imageMap.find(c) == imageMap.end())
            {
                continue;
            }
            imageMap_boosted.emplace(c, std::make_pair(cv::Mat(), cv::Mat()));
            imgROIs.emplace(c, cv::Rect());
            camImgIdxsSearchAddKp.emplace_back(c);
        }

        if (!camImgIdxsSearchAddKp.empty())
        {
            std::vector<std::thread> threads;
            const unsigned imgsCount = camImgIdxsSearchAddKp.size();
    #if DBG_SHOW_ADDITIONAL_KP_GEN
            unsigned threadCount = 1;
    #else
            unsigned threadCount = std::min(imgsCount, static_cast<unsigned>(cpuCnt));
    #endif
            unsigned batchSize = std::ceil(imgsCount / static_cast<float>(threadCount));
            getThreadBatchSize(imgsCount, threadCount, batchSize);

            for (unsigned int i = 0; i < threadCount; ++i)
            {
                const int startIdx = i * batchSize;
                const int endIdx = std::min((i + 1u) * batchSize, imgsCount);
                threads.push_back(std::thread(std::bind(&Matching::prepareImgsLessTextureThreadFunc, this, startIdx, endIdx, std::ref(camImgIdxsSearchAddKp), std::ref(imageMap_boosted), std::ref(imgROIs))));
            }

            for (auto &t : threads)
            {
                if (t.joinable())
                {
                    t.join();
                }
            }

            //Remove empty entries
            std::vector<int> camIdxsSearchAddKp_filtered;
            size_t detect_area = 0;
            for (const auto &c : camImgIdxsSearchAddKp)
            {
                if (imageMap_boosted.at(c).first.empty()){
                    imgROIs.erase(c);
                    imageMap_boosted.erase(c);
                }else{
                    camIdxsSearchAddKp_filtered.emplace_back(c);
                    detect_area += static_cast<size_t>(cv::countNonZero(imageMap_boosted.at(c).second));
                }
            }
            if (camIdxsSearchAddKp_filtered.empty())
            {
                return false;
            }
            detect_area /= camIdxsSearchAddKp_filtered.size();
            cv::Size imgSi = imageMap.begin()->second.first.size();
            double area_ratio = static_cast<double>(detect_area) / static_cast<double>(imgSi.area());
            int nrKeyPointsMaxFull = getMaxNrFeatures(affineInvariantUsed);
            int nr_kp_max = max(static_cast<int>(std::round(static_cast<double>(nrKeyPointsMaxFull) * area_ratio)), 2000);

            //Extract keypoints
            std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> keypoints_descriptors_new;
            std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> init_keypoints_new;
            std::unordered_map<int, std::vector<cv::KeyPoint>> keypoints_combined_new;

            bool res = computeFeaturesOtherImgs(imageMap_boosted, camIdxsSearchAddKp_filtered, keypoints_descriptors_new, &init_keypoints_new, &keypoints_combined_new, nr_kp_max, true);
            if(!res){
                return false;
            }

            //Remove keypoints from masked areas and adapt keypoint positions based on ROIs
            for(const auto &ci : camIdxsSearchAddKp_filtered){
                const cv::Rect &roi = imgROIs.at(ci);
                const cv::Point2f roi_add(static_cast<float>(roi.x), static_cast<float>(roi.y));
                const cv::Mat mask = imageMap_boosted.at(ci).second;

                std::vector<cv::KeyPoint> kps_filtered;
                cv::Mat descr_filtered;
                std::pair<std::vector<cv::KeyPoint>, cv::Mat> &features_ci = keypoints_descriptors_new.at(ci);
                std::vector<cv::KeyPoint> &kps = features_ci.first;
                cv::Mat &descr = features_ci.second;
                for (int i = 0; i < descr.rows; i++)
                {
                    cv::KeyPoint kp = kps.at(i);
                    int x = static_cast<int>(round(kp.pt.x));
                    int y = static_cast<int>(round(kp.pt.y));
                    if(mask.at<unsigned char>(y, x))
                    {
                        kp.pt += roi_add;
                        kps_filtered.emplace_back(move(kp));
                        descr_filtered.push_back(descr.row(i));
                    }
                }
                if (!kps_filtered.empty())
                {
                    kps = move(kps_filtered);
                    descr_filtered.copyTo(descr);
                }
                else{
                    keypoints_descriptors_new.erase(ci);
                    auto kp_init_it = init_keypoints_new.begin();
                    while (kp_init_it != init_keypoints_new.end())
                    {
                        kp_init_it->second.erase(ci);
                        if (kp_init_it->second.empty())
                        {
                            kp_init_it = init_keypoints_new.erase(kp_init_it);
                        }
                        else
                        {
                            kp_init_it++;
                        }
                    }
                    keypoints_combined_new.erase(ci);
                    continue;
                }

                auto kp_init_it = init_keypoints_new.begin();
                while (kp_init_it != init_keypoints_new.end())
                {
                    std::vector<cv::KeyPoint> &kps_init = kp_init_it->second.at(ci);
                    std::vector<cv::KeyPoint> kps_init_filtered;
                    for (size_t i = 0; i < kps_init.size(); i++)
                    {
                        cv::KeyPoint kp = kps_init.at(i);
                        int x = static_cast<int>(round(kp.pt.x));
                        int y = static_cast<int>(round(kp.pt.y));
                        if (mask.at<unsigned char>(y, x))
                        {
                            kp.pt += roi_add;
                            kps_init_filtered.emplace_back(move(kp));
                        }
                    }
                    if (!kps_init_filtered.empty())
                    {
                        kps_init = move(kps_init_filtered);
                        kp_init_it++;
                    }
                    else
                    {
                        kp_init_it->second.erase(ci);
                        if (kp_init_it->second.empty())
                        {
                            kp_init_it = init_keypoints_new.erase(kp_init_it);
                        }
                        else{
                            kp_init_it++;
                        }
                    }
                }

                std::vector<cv::KeyPoint> &kps_comb = keypoints_combined_new.at(ci);
                std::vector<cv::KeyPoint> kps_comb_filtered;
                for (size_t i = 0; i < kps_comb.size(); i++)
                {
                    cv::KeyPoint kp = kps_comb.at(i);
                    int x = static_cast<int>(round(kp.pt.x));
                    int y = static_cast<int>(round(kp.pt.y));
                    if (mask.at<unsigned char>(y, x))
                    {
                        kp.pt += roi_add;
                        kps_comb_filtered.emplace_back(move(kp));
                    }
                }
                if (!kps_comb_filtered.empty())
                {
                    kps_comb = move(kps_comb_filtered);
                }
                else
                {
                    keypoints_combined_new.erase(ci);
                }
            }

            //Add keypoints to global (class) data structures
            for (const auto &kp_descr : keypoints_descriptors_new)
            {
                std::pair<std::vector<cv::KeyPoint>, cv::Mat> &features_ci = keypoints_descriptors.at(kp_descr.first);
                features_ci.first.insert(features_ci.first.end(), kp_descr.second.first.begin(), kp_descr.second.first.end());
                features_ci.second.push_back(kp_descr.second.second);
            }
            for (const auto &kp_type : init_keypoints_new)
            {
                std::unordered_map<int, std::vector<cv::KeyPoint>> &kps_type = init_keypoints.at(kp_type.first);
                for (const auto &kps : kp_type.second){
                    std::vector<cv::KeyPoint> &kps_ci = kps_type.at(kps.first);
                    kps_ci.insert(kps_ci.end(), kps.second.begin(), kps.second.end());
                }
            }
            for (const auto &kp_comb : keypoints_combined_new)
            {
                std::vector<cv::KeyPoint> &kps_ci = keypoints_combined.at(kp_comb.first);
                kps_ci.insert(kps_ci.end(), kp_comb.second.begin(), kp_comb.second.end());
            }
        }
        auto kpadd_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
        cout << "Search for additional keypoints in uniform image regions took " << kpadd_time_ms / 1e3 << " seconds." << endl;

        return true;
    }

    void Matching::prepareImgsLessTextureThreadFunc(const int startIdx, 
                                                    const int endIdx, 
                                                    const std::vector<int> &camImgIdxsSearchAddKp, 
                                                    std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_boosted, 
                                                    std::unordered_map<int, cv::Rect> &imgROIs)
    {
        for (int i = startIdx; i < endIdx; i++)
        {
            const int &ci1 = camImgIdxsSearchAddKp.at(i);
            const cv::Size imgSi = imageMap.at(ci1).first.size();
            const int maxImgSi = max(imgSi.width, imgSi.height);
            const int kernelRadius = max(maxImgSi / 70, 5);
            const int kernelSize = 2 * kernelRadius + 1;
            const cv::Mat maskm = imageMap.at(ci1).second;
            cv::Mat maskkp = cv::Mat::zeros(imgSi, CV_8UC1);
            int nr_kp_in_mask = 0;
            //Generate mask with keypoint positions in masked areas of the motion mask
            for (const auto &kp : keypoints_descriptors.at(ci1).first)
            {
                const cv::Point2f &pt = kp.pt;
                int x = static_cast<int>(round(pt.x));
                int y = static_cast<int>(round(pt.y));
                if (maskm.empty() || maskm.at<unsigned char>(y, x))
                {
                    maskkp.at<unsigned char>(y, x) = 255;
                    nr_kp_in_mask++;
                }
            }

            // cv::namedWindow("mask_kp", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
            // cv::imshow("mask_kp", maskkp);
            // cv::waitKey();
            // cv::destroyWindow("mask_kp");

            cv::Rect bound_out(0, 0, imgSi.width, imgSi.height);
            cv::Mat mask_outer;
            cv::Mat maskm_bound, img_bound;
            //Get an image ROI on which to work on
            if (nr_kp_in_mask > 50)
            {
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize), cv::Point(kernelRadius, kernelRadius));
                cv::morphologyEx(maskkp, maskkp, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);
                int mask_area0 = 0;
                if(maskm.empty()){
                    mask_area0 = imgSi.area();
                }else{
                    mask_area0 = cv::countNonZero(maskm);
                }
                int mask_area_kp = cv::countNonZero(maskkp);
                float ratio0 = 1.f - static_cast<float>(mask_area_kp) / static_cast<float>(mask_area0);
                if (ratio0 < 0.33)
                {
                    continue;
                }
                if(!maskm.empty()){
                    cv::bitwise_xor(maskkp, maskm, mask_outer);
                    cv::bitwise_and(mask_outer, maskm, mask_outer);
                }else{
                    cv::bitwise_not(maskkp, mask_outer);
                }

                //Remove small masked areas
                cv::Mat labels, stats, centroids;
                cv::connectedComponentsWithStats(mask_outer, labels, stats, centroids);
                if (stats.rows > 5)
                {
                    std::set<int, std::greater<int>> comp_areas;
                    for (int i = 1; i < stats.rows; i++)
                    {
                        comp_areas.emplace(stats.at<int>(cv::Point(cv::CC_STAT_AREA, i)));
                    }
                    int area_cnt = 0, area_last = 0;
                    for(const auto &a: comp_areas){
                        area_cnt++;
                        if (area_cnt < 2)
                        {
                            area_last = a;
                            continue;
                        }

                        float redu_fac = static_cast<float>(a) / static_cast<float>(area_last);
                        if (redu_fac < 0.4f){
                            break;
                        }
                        area_last = a;
                        if (area_cnt > 20)
                        {
                            break;
                        }
                    }
                    if (area_cnt < 20 && static_cast<size_t>(area_cnt) < comp_areas.size())
                    {
                        const int area_min = max(area_last / 10, 300);
                        cv::Mat mask_filtered = mask_outer.clone();
                        for (int i = 1; i < stats.rows; i++)
                        {
                            int area = stats.at<int>(cv::Point(cv::CC_STAT_AREA, i));
                            if (area < area_last)
                            {
                                int x_lu = stats.at<int>(cv::Point(cv::CC_STAT_LEFT, i));
                                int y_lu = stats.at<int>(cv::Point(cv::CC_STAT_TOP, i));
                                int w = stats.at<int>(cv::Point(cv::CC_STAT_WIDTH, i));
                                int h = stats.at<int>(cv::Point(cv::CC_STAT_HEIGHT, i));
                                for (int y = y_lu; y < y_lu + h; y++)
                                {
                                    for (int x = x_lu; x < x_lu + w; x++)
                                    {
                                        if (labels.at<int>(y, x) == i)
                                        {
                                            mask_filtered.at<unsigned char>(y, x) = 0;
                                            if (area < area_min)
                                            {
                                                mask_outer.at<unsigned char>(y, x) = 0;
                                            }
                                        }
                                    }
                                }
                            }
                        }
    #if DBG_SHOW_ADDITIONAL_KP_GEN == 1
                        cv::namedWindow("mask_xor_orig", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
                        cv::namedWindow("mask_xor_filter", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
                        cv::imshow("mask_xor_orig", mask_outer);
                        cv::imshow("mask_xor_filter", mask_filtered);
                        cv::waitKey();
                        cv::destroyWindow("mask_xor_orig");
                        cv::destroyWindow("mask_xor_filter");
    #endif
                        bound_out = cv::boundingRect(mask_filtered);
                    }
                    else{
                        bound_out = cv::boundingRect(mask_outer);
                    }
                }else{
                    bound_out = cv::boundingRect(mask_outer);
                }
                mask_outer = mask_outer(bound_out);
                if(imageMap.at(ci1).second.empty()){
                    maskm_bound = cv::Mat::ones(bound_out.size(), CV_8UC1);
                }else{
                    maskm_bound = imageMap.at(ci1).second(bound_out).clone();
                }
                img_bound = imageMap.at(ci1).first(bound_out).clone();
            }else{
                if(imageMap.at(ci1).second.empty()){
                    maskm_bound = cv::Mat::ones(imageMap.at(ci1).first.size(), CV_8UC1);
                }else{
                    maskm_bound = imageMap.at(ci1).second.clone();
                }
                img_bound = imageMap.at(ci1).first.clone();
                mask_outer = maskm_bound;
            }
    #if DBG_SHOW_ADDITIONAL_KP_GEN == 1
            cv::namedWindow("mask_orig_bound", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
            // cv::namedWindow("mask_kp_bound", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
            cv::namedWindow("mask_xor_bound", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
            cv::imshow("mask_orig_bound", maskm_bound);
            cv::imshow("mask_xor_bound", mask_outer);
            cv::waitKey();
            cv::destroyWindow("mask_orig_bound");
            cv::destroyWindow("mask_xor_bound");
    #endif

    #if DBG_SHOW_ADDITIONAL_KP_GEN
            cv::namedWindow("highlight", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
    #if DBG_SHOW_ADDITIONAL_KP_GEN == 2
            cv::namedWindow("parameters");
    #endif
            cv::namedWindow("input", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
            // cv::namedWindow("other", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
            // cv::namedWindow("other2", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
            cv::imshow("input", imageMap.at(ci1).first);
    #endif

            float shadow_amount_percent = 0.8f, shadow_tone_percent = 0.8f, highlight_amount_percent = 0.66f, highlight_tone_percent = 0.8f;
            int shadow_radius = 5, highlight_radius = 5;
            cv::Mat highlight_img;
    #if DBG_SHOW_ADDITIONAL_KP_GEN == 2
            int shadow_amount_percent_int = 80, shadow_tone_percent_int = 80, highlight_amount_percent_int = 66, highlight_tone_percent_int = 80;

            cv::createTrackbar("shadow amount percent", "parameters", &shadow_amount_percent_int, 100);
            cv::createTrackbar("shadow tone percent", "parameters", &shadow_tone_percent_int, 100);
            cv::createTrackbar("highlight amount percent", "parameters", &highlight_amount_percent_int, 100);
            cv::createTrackbar("highlight tone percent", "parameters", &highlight_tone_percent_int, 100);
            cv::createTrackbar("shadow radius", "parameters", &shadow_radius, 50);
            cv::createTrackbar("highlight radius", "parameters", &highlight_radius, 50);

            while(true)
            {
                shadow_amount_percent = static_cast<float>(shadow_amount_percent_int) / 100.f;
                shadow_tone_percent = static_cast<float>(shadow_tone_percent_int) / 100.f;
                highlight_amount_percent = static_cast<float>(highlight_amount_percent_int) / 100.f;
                highlight_tone_percent = static_cast<float>(highlight_tone_percent_int) / 100.f;
    #endif

                highlight_img = shadowHighlightCorrection(img_bound, shadow_amount_percent, shadow_tone_percent, shadow_radius + 1, highlight_amount_percent, highlight_tone_percent, highlight_radius + 1, true);

    #if DBG_SHOW_ADDITIONAL_KP_GEN
                cv::imshow("highlight", highlight_img);
    #endif
    #if DBG_SHOW_ADDITIONAL_KP_GEN == 2
                char key = (char)cv::waitKey(30);
                if (key == 'q' || key == 27)
                {
                    break;
                }
            }
    #elif DBG_SHOW_ADDITIONAL_KP_GEN == 1
            cv::waitKey();
    #endif
    #if DBG_SHOW_ADDITIONAL_KP_GEN
            cv::destroyAllWindows();
    #endif
    #if DBG_SHOW_ADDITIONAL_KP_GEN == 2
            cout << "shadow amount: " << shadow_amount_percent_int << endl;
            cout << "shadow tone: " << shadow_tone_percent_int << endl;
            cout << "shadow radius: " << shadow_radius + 1 << endl;
            cout << "highlight amount: " << highlight_amount_percent_int << endl;
            cout << "highlight tone: " << highlight_tone_percent_int << endl;
            cout << "highlight radius: " << highlight_radius + 1 << endl;
    #endif

            //Blur unmasked image regions
            cv::Mat blurred_img, blurred_mask, partly_blurred;
            int kernelRadius2 = min(kernelRadius, 12);
            int kernelSize2 = 2 * kernelRadius2 + 1;
            cv::blur(highlight_img, blurred_img, cv::Size(kernelSize2, kernelSize2));
            cv::blur(mask_outer, blurred_mask, cv::Size(kernelSize2, kernelSize2));

            cv::Mat blurred_mask1, blurred_mask1_inv;
            blurred_mask.convertTo(blurred_mask1, CV_32FC1, 1.0 / 255.0);
            blurred_mask1_inv = 1.f - blurred_mask1;

            cv::Mat blurred_img_flt, highlight_img_flt;
            blurred_img.convertTo(blurred_img_flt, CV_32FC1);
            highlight_img.convertTo(highlight_img_flt, CV_32FC1);

            partly_blurred = blurred_img_flt.mul(blurred_mask1_inv);
            partly_blurred += highlight_img_flt.mul(blurred_mask1);
            cv::convertScaleAbs(partly_blurred, partly_blurred);
#if DBG_SHOW_ADDITIONAL_KP_GEN == 1
            cv::namedWindow("partly_blurred", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
            cv::imshow("partly_blurred", partly_blurred);
            cv::waitKey();
            cv::destroyWindow("partly_blurred");
#endif
            //Store results
            imgROIs.at(ci1) = bound_out;
            std::pair<cv::Mat, cv::Mat> &res = imageMap_boosted.at(ci1);
            partly_blurred.copyTo(res.first);
            mask_outer.copyTo(res.second);
        }
    }

    void Matching::getImgsAndMasks(std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &images){
        images = imageMap;
    }

    void Matching::moveImgsAndMasks(std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &images){
        images = move(imageMap);
    }

    std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &Matching::getImgsAndMasksRef()
    {
        return imageMap;
    }

    cv::Size Matching::getImgSize() const
    {
        return imageMap.begin()->second.first.size();
    }

    int Matching::getNrCams()
    {
        return largest_cam_idx;
    }

    void Matching::getRawKeypoints(std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &raw_keypoints)
    {
        raw_keypoints = init_keypoints;
    }

    void Matching::moveRawKeypoints(std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &raw_keypoints)
    {
        raw_keypoints = move(init_keypoints);
    }

    std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &Matching::getRawKeypointsRef()
    {
        return init_keypoints;
    }

    void Matching::getCombinedKeypoints(std::unordered_map<int, std::vector<cv::KeyPoint>> &kpts_combined)
    {
        kpts_combined = keypoints_combined;
    }

    void Matching::moveCombinedKeypoints(std::unordered_map<int, std::vector<cv::KeyPoint>> &kpts_combined)
    {
        kpts_combined = move(keypoints_combined);
    }

    std::unordered_map<int, std::vector<cv::KeyPoint>> &Matching::getCombinedKeypointsRef()
    {
        return keypoints_combined;
    }

    void Matching::getFinalFeatures(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features)
    {
        features = keypoints_descriptors;
    }

    void Matching::moveFinalFeatures(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features)
    {
        features = move(keypoints_descriptors);
    }

    std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &Matching::getFinalFeaturesRef()
    {
        return keypoints_descriptors;
    }

    void Matching::getFinalMatches(std::unordered_map<std::pair<int, int>, MatchData, pair_hash, pair_EqualTo> &matches)
    {
        matches = matches_filt;
    }

    void Matching::moveFinalMatches(std::unordered_map<std::pair<int, int>, MatchData, pair_hash, pair_EqualTo> &matches)
    {
        matches = move(matches_filt);
    }

    std::unordered_map<std::pair<int, int>, MatchData, pair_hash, pair_EqualTo> &Matching::getFinalMatchesRef()
    {
        return matches_filt;
    }

    void Matching::getCamPairs(std::vector<std::pair<int, int>> &pair_indices)
    {
        pair_indices = cam_pair_idx;
    }

    std::vector<std::pair<int, int>> &Matching::getCamPairs()
    {
        return cam_pair_idx;
    }

    void Matching::getFullMatchingData(MatchDataCams &data)
    {
        for (const auto &m : imageMap)
        {
            data.masks.emplace(m.first, m.second.second.clone());
            data.images.emplace(m.first, m.second.first.clone());
        }
        data.features = keypoints_descriptors;
        data.matches = matches_filt;
        data.cam_pair_indices = cam_pair_idx;
        data.nr_cameras = largest_cam_idx;
        data.imgScale = img_scaling;
    }

    void Matching::moveFullMatchingData(MatchDataCams &data)
    {
        for (const auto &m : imageMap)
        {
            data.masks.emplace(m.first, move(m.second.second));
            data.images.emplace(m.first, move(m.second.first));
        }
        data.features = move(keypoints_descriptors);
        data.matches = move(matches_filt);
        data.cam_pair_indices = move(cam_pair_idx);
        data.nr_cameras = move(largest_cam_idx);
        data.imgScale = move(img_scaling);
    }

    MatchDataCams Matching::getFullMatchingData()
    {
        MatchDataCams tmp;
        for (auto &m : imageMap)
        {
            tmp.masks.emplace(m.first, m.second.second);
            tmp.images.emplace(m.first, m.second.first);
        }
        tmp.features = keypoints_descriptors;
        tmp.matches = matches_filt;
        tmp.cam_pair_indices = cam_pair_idx;
        tmp.nr_cameras = largest_cam_idx;
        tmp.imgScale = img_scaling;
        return tmp;
    }

    std::shared_ptr<MatchDataCams> Matching::getFullMatchingDataPtr()
    {
        return std::make_shared<MatchDataCams>(imageMap, keypoints_descriptors, matches_filt, cam_pair_idx, largest_cam_idx, img_scaling);
    }

    MatchDataCams Matching::moveFullMatchingData()
    {
        MatchDataCams tmp;
        for (auto &m : imageMap)
        {
            tmp.masks.emplace(m.first, move(m.second.second));
            tmp.images.emplace(m.first, move(m.second.first));
        }
        tmp.features = move(keypoints_descriptors);
        tmp.matches = move(matches_filt);
        tmp.cam_pair_indices = move(cam_pair_idx);
        tmp.nr_cameras = move(largest_cam_idx);
        tmp.imgScale = move(img_scaling);
        return tmp;
    }

    void Matching::writeBinary(const std::string &filename) const
    {
        ofstream resultsToFile(filename, ios::out | ios::binary);
        if (!resultsToFile.is_open())
        {
            throw runtime_error("Error creating output file for writing results.");
        }

        resultsToFile.write((char *)&largest_cam_idx, sizeof(largest_cam_idx));
        resultsToFile.write((char *)&img_scaling, sizeof(img_scaling));
        const cv::Size imgSi = getImgSize();
        resultsToFile.write((char *)&imgSi, sizeof(imgSi));

        size_t nr_values = cam_pair_idx.size();
        resultsToFile.write((char *)&nr_values, sizeof(nr_values));
        for (const auto &v : cam_pair_idx)
        {
            resultsToFile.write((char *)&v, sizeof(v));
        }

        nr_values = matches_filt.size();
        resultsToFile.write((char *)&nr_values, sizeof(nr_values));
        for (const auto &v : matches_filt)
        {
            resultsToFile.write((char *)&v.first, sizeof(v.first));
            
            const MatchData &data = v.second;
            nr_values = data.matches.size();
            resultsToFile.write((char *)&nr_values, sizeof(nr_values));
            for (const auto &m : data.matches)
            {
                FileHelper::matchToBinary(resultsToFile, m);
            }

            FileHelper::keypointsToBinary(resultsToFile, data.kps1);
            FileHelper::keypointsToBinary(resultsToFile, data.kps2);

            FileHelper::cvMatToBinary(resultsToFile, data.descr1);
            FileHelper::cvMatToBinary(resultsToFile, data.descr2);

            bool mask_nempty = !data.inlier_mask.empty();
            resultsToFile.write((char *)&mask_nempty, sizeof(mask_nempty));
            if (mask_nempty)
            {
                FileHelper::cvMatToBinary(resultsToFile, data.inlier_mask);
            }

            resultsToFile.write((char *)&data.used_cnt, sizeof(data.used_cnt));
        }

        nr_values = keypoints_descriptors.size();
        resultsToFile.write((char *)&nr_values, sizeof(nr_values));
        for (const auto &f : keypoints_descriptors)
        {
            resultsToFile.write((char *)&f.first, sizeof(f.first));
            FileHelper::keypointsToBinary(resultsToFile, f.second.first);
            FileHelper::cvMatToBinary(resultsToFile, f.second.second);
        }

        resultsToFile.close();
    }

    void Matching::readBinary(const std::string &filename)
    {
        ifstream resultsFromFile(filename, ios::in | ios::binary);
        if (!resultsFromFile.is_open())
        {
            throw runtime_error("Error opening binary file for reading matching results.");
        }

        resultsFromFile.read((char *)&largest_cam_idx, sizeof(int));
        double img_scaling_read;
        resultsFromFile.read((char *)&img_scaling_read, sizeof(double));
        if (!nearZero(img_scaling_read - img_scaling)){
            throw runtime_error("Image scaling factors specified and read from file do not match");
        }
        const cv::Size imgSi = getImgSize();
        cv::Size imgSi_read;
        resultsFromFile.read((char *)&imgSi_read, sizeof(cv::Size));
        if (imgSi_read.width != imgSi.width || imgSi_read.height != imgSi.height)
        {
            throw runtime_error("Image size of provided images and read from file do not match");
        }

        size_t nr_values;
        resultsFromFile.read((char *)&nr_values, sizeof(size_t));
        cam_pair_idx.clear();
        for (size_t i = 0; i < nr_values; i++)
        {
            pair<int, int> tmp;
            resultsFromFile.read((char *)&tmp, sizeof(pair<int, int>));
            cam_pair_idx.emplace_back(move(tmp));
        }

        resultsFromFile.read((char *)&nr_values, sizeof(size_t));
        matches_filt.clear();
        for (size_t i = 0; i < nr_values; i++)
        {
            MatchData data;
            pair<int, int> idx;
            resultsFromFile.read((char *)&idx, sizeof(pair<int, int>));

            size_t nr_values1;
            resultsFromFile.read((char *)&nr_values1, sizeof(size_t));

            for (size_t j = 0; j < nr_values1; j++)
            {
                data.matches.emplace_back(FileHelper::matchFromBinary(resultsFromFile));
            }

            FileHelper::keypointsFromBinary(resultsFromFile, data.kps1);
            FileHelper::keypointsFromBinary(resultsFromFile, data.kps2);

            data.descr1 = FileHelper::cvMatFromBinary(resultsFromFile);
            data.descr2 = FileHelper::cvMatFromBinary(resultsFromFile);

            bool mask_nempty;
            resultsFromFile.read((char *)&mask_nempty, sizeof(bool));
            if (mask_nempty)
            {
                data.inlier_mask = FileHelper::cvMatFromBinary(resultsFromFile);
            }

            resultsFromFile.read((char *)&data.used_cnt, sizeof(int));

            matches_filt.emplace(idx, move(data));
        }

        resultsFromFile.read((char *)&nr_values, sizeof(size_t));
        keypoints_descriptors.clear();
        for (size_t i = 0; i < nr_values; i++)
        {
            int idx;
            resultsFromFile.read((char *)&idx, sizeof(int));

            std::vector<cv::KeyPoint> kps;
            FileHelper::keypointsFromBinary(resultsFromFile, kps);

            cv::Mat descr = FileHelper::cvMatFromBinary(resultsFromFile);

            keypoints_descriptors.emplace(idx, make_pair(move(kps), descr));
        }
        resultsFromFile.close();
    }

#if DBG_SHOW_MATCHES
    void drawMatchesImgPair(const cv::Mat &img1, 
                            const cv::Mat &img2, 
                            const std::vector<cv::KeyPoint> &kp1, 
                            const std::vector<cv::KeyPoint> &kp2, 
                            const std::vector<cv::DMatch> &matches, 
                            const int &img1_idx, 
                            const int &img2_idx, 
                            const size_t &nrMatchesLimit)
    {
        const int maxImgWidth = 700;
        double s = static_cast<double>(maxImgWidth) / static_cast<double>(img1.cols);
        const float s_f = static_cast<float>(s);
        cv::Mat img_s1, img_s2;
        cv::resize(img1, img_s1, cv::Size(), s, s, cv::INTER_CUBIC);
        cv::resize(img2, img_s2, cv::Size(), s, s, cv::INTER_CUBIC);
        std::vector<cv::DMatch> matches_s;
        std::vector<cv::KeyPoint> kp1_s, kp2_s;
        int idx = 0;
        for (size_t i = 0; i < matches.size(); i++)
        {
            cv::DMatch m = matches.at(i);
            cv::KeyPoint pt1 = kp1.at(m.queryIdx);
            cv::KeyPoint pt2 = kp2.at(m.trainIdx);
            pt1.size *= s_f;
            pt1.pt *= s_f;
            kp1_s.emplace_back(move(pt1));
            pt2.size *= s_f;
            pt2.pt *= s_f;
            kp2_s.emplace_back(move(pt2));
            m.queryIdx = idx;
            m.trainIdx = idx;
            matches_s.emplace_back(move(m));
            idx++;
        }

        std::vector<cv::DMatch> matches_s_redu;
        if (nrMatchesLimit > 0 && matches_s.size() > nrMatchesLimit)
        {
            std::vector<size_t> vec_idx(matches_s.size());
            std::iota(vec_idx.begin(), vec_idx.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());

            std::shuffle(vec_idx.begin(), vec_idx.end(), g);
            for (size_t i = 0; i < nrMatchesLimit; i++)
            {
                matches_s_redu.emplace_back(matches_s.at(vec_idx.at(i)));
            }
        }
        else
        {
            matches_s_redu = matches_s;
        }

        cv::Mat outImg;
        cv::drawMatches(img_s1, kp1_s, img_s2, kp2_s, matches_s_redu, outImg);
        const int textHeight = 12;
        const double fontScale = cv::getFontScaleFromHeight(cv::FONT_HERSHEY_SIMPLEX, textHeight);
        string text1 = "cam" + std::to_string(img1_idx);
        cv::Point text_bl1(5, textHeight + 3);
        cv::putText(outImg, text1, text_bl1, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 255));
        
        string text2 = "cam" + std::to_string(img2_idx);
        cv::Point text_bl2(img_s1.cols + 5, textHeight + 3);
        cv::putText(outImg, text2, text_bl2, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 255));

        cv::namedWindow("feature_matches", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::imshow("feature_matches", outImg);
        cv::waitKey(0);
        cv::destroyWindow("feature_matches");
    }
#endif

#ifdef WITH_AKAZE_CUDA
    cv::Ptr<cv::cuda::Feature2DAsync> createCudaDetector(const string &keypointtype, const int limitNrfeatures)
    {
        cv::Ptr<cv::cuda::Feature2DAsync> detector;

        if (!keypointtype.compare("ORB"))
        {
            detector = cv::cuda::ORB::create(limitNrfeatures);
        }
        else
        {
            throw std::runtime_error("GPU keypoint type not supported.");
        }

        return detector;
    }

    cv::Ptr<cv::cuda::Feature2DAsync> createCudaExtractor(std::string const &descriptortype, const int &nrFeaturesMax)
    {
        cv::Ptr<cv::cuda::Feature2DAsync> extractor;

        if (!descriptortype.compare("ORB"))
        {
            extractor = cv::cuda::ORB::create(nrFeaturesMax);
        }
        else
        {
            throw std::runtime_error("Descriptor type not supported.");
        }

        return extractor;
    }

    bool IsFeatureCudaTypeSupported(const std::string &type)
    {
        std::vector<std::string> vecSupportedTypes = GetSupportedFeatureCudaTypes();

        if (std::find(vecSupportedTypes.begin(), vecSupportedTypes.end(), type) != vecSupportedTypes.end())
        {
            return true;
        }

        return false;
    }

    std::vector<std::string> GetSupportedFeatureCudaTypes()
    {
        int const nrSupportedTypes = 2;

        static std::string types[] = { "ORB", "AKAZE" };
        return std::vector<std::string>(types, types + nrSupportedTypes);
    }
#endif
#if DBG_SHOW_KEYPOINTS || DBG_SHOW_KP_FILT_MASK || DBG_SHOW_KEYPOINTS_FILTERED || DBG_SHOW_ADDITIONAL_KP_GEN
    void visualizeKeypoints(const std::unordered_map<int, std::vector<cv::KeyPoint>> &keypoints_combined, 
                            const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap,
                            const std::string &feature_type)
    {
        for(const auto &img : imageMap){
            const std::string img_name = to_string(img.first);
            visualizeKeypoints(img.second.first, keypoints_combined.at(img.first), img_name, feature_type);
        }
    }

    void visualizeKeypoints(const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors, 
                            const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap, 
                            const std::string &feature_type)
    {
        for (const auto &img : imageMap)
        {
            const std::string img_name = to_string(img.first);
            visualizeKeypoints(img.second.first, keypoints_descriptors.at(img.first).first, img_name, feature_type);
        }
    }

    void visualizeKeypoints(const std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &init_keypoints, 
                            const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap, 
                            const std::string &descriptor_type)
    {
        for (const auto &kp1 : init_keypoints)
        {
            std::string feature_type = "init_" + kp1.first + "_" + descriptor_type;
            visualizeKeypoints(kp1.second, imageMap, feature_type);
        }
    }

    void visualizeKeypoints(const cv::Mat &img, 
                            const std::vector<cv::KeyPoint> &keypoints, 
                            const std::string &img_name, 
                            const std::string &feature_type)
    {
        const int maxImgDim = max(img.cols, img.rows);
        cv::Mat img_s;
        double s = 1.0;
        if (maxImgDim > DBG_SHOW_KEYPOINTS_IMG_SIZE_MAX)
        {
            s = static_cast<double>(DBG_SHOW_KEYPOINTS_IMG_SIZE_MAX) / static_cast<double>(maxImgDim);
            cv::resize(img, img_s, cv::Size(), s, s, cv::INTER_CUBIC);
        }
        else{
            img_s = img;
        }
        const float sf = static_cast<float>(s);

        cv::Mat color;
        cv::cvtColor(img_s, color, cv::COLOR_GRAY2BGR);
        const cv::Vec3b pix_val(0, 0, 255);
        for (const auto &kp : keypoints)
        {
            const cv::Point pt(static_cast<int>(round(sf * kp.pt.x)), static_cast<int>(sf * round(kp.pt.y)));
            if (maxImgDim > DBG_SHOW_KEYPOINTS_IMG_SIZE_MAX){
                cv::circle(color, pt, 2, pix_val, cv::FILLED);
            }else{
                color.at<cv::Vec3b>(pt) = pix_val;
            }
        }
        const std::string window_name = "keypoints_" + img_name + "_" + feature_type;
        cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::imshow(window_name, color);
        cv::waitKey(0);
        cv::destroyWindow(window_name);
    }
#endif

#if DBG_SHOW_KP_FILT_MASK
    void visualizeImg(const cv::Mat &img, const std::string &img_baseName, const int img_idx)
    {
        const std::string window_name = img_baseName + "_" + to_string(img_idx);
        cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::imshow(img_baseName, img);
        cv::waitKey(0);
        cv::destroyWindow(img_baseName);
    }
#endif

} // namepace matchinglib
