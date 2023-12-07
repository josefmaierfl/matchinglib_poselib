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
 FILE: correspondences.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for generating matched feature sets out of image
              information.
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "matchinglib/glob_includes.h"
#include "matchinglib/matching_structs.h"
#include "matchinglib/trees.h"

#include "matchinglib/matchinglib_api.h"
#include <string>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <mutex>

namespace matchinglib
{

/* --------------------------- Defines --------------------------- */


/* --------------------- Function prototypes --------------------- */

// Generation of features followed by matching, filtering, and subpixel-refinement.
int MATCHINGLIB_API getCorrespondences(cv::Mat &img1,
                                       cv::Mat &img2,
                                       std::vector<cv::DMatch> &finalMatches,
                                       std::vector<cv::KeyPoint> &kp1,
                                       std::vector<cv::KeyPoint> &kp2,
                                       std::string featuretype = "FAST",
                                       std::string extractortype = "FREAK",
                                       std::string matchertype = "GMBSOF",
                                       bool dynamicKeypDet = true,
                                       int limitNrfeatures = 8000,
                                       bool VFCrefine = false,
                                       bool GMSrefine = false,
                                       bool ratioTest = true,
                                       bool SOFrefine = false,
                                       int subPixRefine = 0,
                                       int verbose = 0,
                                       std::string idxPars_NMSLIB = "",
                                       std::string queryPars_NMSLIB = "");

//Generation of features followed by matching, filtering, and subpixel-refinement.
int MATCHINGLIB_API getCorrespondences(cv::Mat &img1,
                                       cv::Mat &img2,
                                       std::vector<cv::DMatch> &finalMatches,
                                       std::vector<cv::KeyPoint> &kp1,
                                       std::vector<cv::KeyPoint> &kp2,
                                       std::mt19937 &mt,
                                       std::string featuretype = "FAST",
                                       std::string extractortype = "FREAK",
                                       std::string matchertype = "GMBSOF",
                                       bool dynamicKeypDet = true,
                                       int limitNrfeatures = 8000,
                                       bool VFCrefine = false,
                                       bool GMSrefine = false,
                                       bool ratioTest = true,
                                       bool SOFrefine = false,
                                       int subPixRefine = 0,
                                       int verbose = 0,
                                       std::string idxPars_NMSLIB = "",
                                       std::string queryPars_NMSLIB = "");

void MATCHINGLIB_API filterMatchesSOF(const std::vector<cv::KeyPoint> &keypoints1,
                          const std::vector<cv::KeyPoint> &keypoints2,
                          const cv::Size &imgSi,
                          std::vector<cv::DMatch> &matches);

// bool MATCHINGLIB_API IsKeypointTypeSupported(std::string const& type);
// std::vector<std::string> MATCHINGLIB_API GetSupportedKeypointTypes();

// bool MATCHINGLIB_API IsDescriptorTypeSupported(std::string const& type);
// std::vector<std::string> MATCHINGLIB_API GetSupportedDescriptorTypes();

bool MATCHINGLIB_API IsDescriptorSupportedByMatcher(const std::string &descriptorType, const std::string &matcherType);

bool MATCHINGLIB_API getMatch3Corrs(const cv::Point2f &pt1, const cv::Point2f &pt2, 
                                    const cv::Mat &F1, const cv::Mat &F2, 
                                    const cv::Mat descr1, const cv::Mat descr2, 
                                    const FeatureKDTree &ft, 
                                    cv::Mat &descr3, 
                                    cv::Point2f &pt3, 
                                    const double &descr_dist_max, 
                                    const float r_sqrd = 200.f);

struct MATCHINGLIB_API AffineMatchesFilterData{
    std::mutex m;
    std::unordered_map<int, std::set<int>> &kp_descr_valid_indices;
    AffineMatchesFilterData(std::unordered_map<int, std::set<int>> &kp_descr_valid_indices_) : kp_descr_valid_indices(kp_descr_valid_indices_){}

    void find_emplace(const int &c);
    void emplaceIdx(const int &c, const int &val);
};

void MATCHINGLIB_API scaleEqualizeImg(const cv::Mat &img_in, cv::Mat &img_out, const double &img_scaling, const bool equalizeImg = true);
// Image Shadow / Highlight Correction
cv::Mat MATCHINGLIB_API shadowHighlightCorrection(cv::InputArray img, 
                                                  const float &shadow_amount_percent, 
                                                  const float &shadow_tone_percent, 
                                                  const int &shadow_radius, 
                                                  const float &highlight_amount_percent, 
                                                  const float &highlight_tone_percent, 
                                                  const int &highlight_radius, 
                                                  const bool histEqual = true);

std::string MATCHINGLIB_API getIDfromImages(const std::vector<cv::Mat> &imgs);
std::string getIDfromImages(const std::unordered_map<int, cv::Mat> &images);
std::string getIDfromImages(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap);

class MATCHINGLIB_API Matching
{
public:
    Matching(const std::vector<std::string> &img_file_names,
             const std::vector<std::string> &keypoint_types, 
             const std::string &descriptor_type, 
             const std::string &matcher_type = "HNSW",
             const std::vector<std::string> &mask_file_names = std::vector<std::string>(),
             const std::vector<int> &img_indices = std::vector<int>(),
             const bool sort_file_names = false,
             const double &img_scale = 0.5, 
             const bool equalizeImgs = true,
             const int &nrFeaturesToExtract_ = 5000,
             const int &cpuCnt_ = -2);
    Matching(const std::vector<cv::Mat> &imgs,
             const std::vector<std::string> &keypoint_types, 
             const std::string &descriptor_type, 
             const std::string &matcher_type = "HNSW",
             const std::vector<cv::Mat> &masks = std::vector<cv::Mat>(),
             const std::vector<int> &img_indices = std::vector<int>(),
             const double &img_scale = 0.5, 
             const bool equalizeImgs = true, 
             const int &nrFeaturesToExtract_ = 5000,
             const int &cpuCnt_ = -2);
    ~Matching();

    bool compute(const bool affineInvariant = false, const bool useCuda = true);
    bool computeKeypointsOnly(const bool useCuda = true);
    bool computeKeypointsOnly(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                              const std::vector<int> &indices_, 
                              std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &init_keypoints_, 
                              std::unordered_map<int, std::vector<cv::KeyPoint>> &keypoints_combined_, 
                              const int &nrKeyPointsMax, 
                              const bool useCuda = true);
    bool getMatches();

    void getImgsAndMasks(std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &images);
    void moveImgsAndMasks(std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &images);
    std::unordered_map<int, std::pair<cv::Mat, cv::Mat>>& getImgsAndMasksRef();

    cv::Size getImgSize() const;
    int getNrCams();

    void getRawKeypoints(std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &raw_keypoints);
    void moveRawKeypoints(std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &raw_keypoints);
    std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &getRawKeypointsRef();

    void getCombinedKeypoints(std::unordered_map<int, std::vector<cv::KeyPoint>> &kpts_combined);
    void moveCombinedKeypoints(std::unordered_map<int, std::vector<cv::KeyPoint>> &kpts_combined);
    std::unordered_map<int, std::vector<cv::KeyPoint>> &getCombinedKeypointsRef();

    void getFinalFeatures(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features);
    void moveFinalFeatures(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features);
    std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &getFinalFeaturesRef();

    void getFinalMatches(std::unordered_map<std::pair<int, int>, MatchData, pair_hash, pair_EqualTo> &matches);
    void moveFinalMatches(std::unordered_map<std::pair<int, int>, MatchData, pair_hash, pair_EqualTo> &matches);
    std::unordered_map<std::pair<int, int>, MatchData, pair_hash, pair_EqualTo> &getFinalMatchesRef();

    void getCamPairs(std::vector<std::pair<int, int>> &pair_indices); //Pair: Cameras forming stereo pairs
    std::vector<std::pair<int, int>> &getCamPairs();

    void getFullMatchingData(MatchDataCams &data);
    void moveFullMatchingData(MatchDataCams &data);
    MatchDataCams getFullMatchingData();
    std::shared_ptr<MatchDataCams> getFullMatchingDataPtr();
    MatchDataCams moveFullMatchingData();

    void writeBinary(const std::string &filename) const;
    void readBinary(const std::string &filename);

private:
    void loadImages();
    void loadImageThreadFunc(const int startIdx, const int endIdx);
    void preprocessImages();
    void preprocessImageThreadFunc(const int startIdx, const int endIdx);

    bool getKeypoints(const bool useCuda = true);
    bool getKeypoints(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                      const std::vector<int> &indices_, 
                      std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &init_keypoints_, 
                      const int &nrKeyPointsMax = 8000, 
                      const bool useCuda = true);
    bool combineKeypoints();
    bool combineKeypoints(std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> &init_keypoints_, 
                          const std::vector<int> &indices_, 
                          std::unordered_map<int, std::vector<cv::KeyPoint>> &keypoints_combined_);

    bool getDescriptors(const bool useCuda = true);
    bool getDescriptors(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                        const std::unordered_map<int, std::vector<cv::KeyPoint>> &keypoints_combined_, 
                        const std::vector<int> &indices_, 
                        std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                        const int &nrKeyPointsMax = 8000, 
                        const bool useCuda = true);

    bool getFeatures(const bool useCuda = true);
    bool getFeatures(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                     const std::vector<int> &indices_, 
                     std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                     std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> *init_keypoints_, 
                     std::unordered_map<int, std::vector<cv::KeyPoint>> *keypoints_combined_, 
                     const int &nrKeyPointsMax = 3000, 
                     const bool useCuda = true);
    bool getFeatures(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                     const std::vector<int> &indices_, 
                     const std::vector<std::pair<std::string, int>> &indices_kp_, 
                     std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                     const int &nrKeyPointsMax = 3000, 
                     const bool useCuda = true);
    bool getFeaturesAffine(const bool useCuda = true);
    bool getFeaturesAffine(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                           const std::vector<int> &indices_, 
                           std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                           std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> *init_keypoints_ = nullptr, 
                           std::unordered_map<int, std::vector<cv::KeyPoint>> *keypoints_combined_ = nullptr, 
                           const int &nrKeyPointsMax = 3000, 
                           const bool useCuda = true);
    bool getFeaturesAffine(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                           const std::vector<int> &indices_, 
                           const std::vector<std::pair<std::string, int>> &indices_kp_, 
                           std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                           const int &nrKeyPointsMax = 3000, 
                           const bool useCuda = true);

    int getMaxNrFeatures(const bool affineInvariant);
    bool computeFeaturesOtherImgs(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                                  const std::vector<int> &indices_, 
                                  std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                                  std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> *init_keypoints_, 
                                  std::unordered_map<int, std::vector<cv::KeyPoint>> *keypoints_combined_, 
                                  const int &nrKeyPointsMax, 
                                  const bool useCuda = true);
    bool computeFeaturesOtherImgs(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_, 
                                  const std::vector<int> &indices_, 
                                  std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors_, 
                                  std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> *init_keypoints_, 
                                  std::unordered_map<int, std::vector<cv::KeyPoint>> *keypoints_combined_, 
                                  const bool useCuda = true);

    bool getMatches(const bool affineInvariant);
    void applyMaskToFeatures(bool invertMask, std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &kp_descr_out);
    void getMatchesThreadFunc(const int startIdx, 
                              const int endIdx, 
                              const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> *kp_descr_in, 
                              std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> *matches_out, 
                              std::exception_ptr &thread_exception, 
                              std::mt19937 mt);
    bool filterMatches(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &kp_descr_in, 
                       std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> &matches_in);

    void filterAreaBasedAffine();
    void filterResponseAffine();
    void filterResponseAffine(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features_map);
    void filterResponseAreaBasedAffine(const int &limitNrFeatures = 12000);
    void filterResponseAreaBasedAffineImpl(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features_map, 
                                           const std::unordered_map<int, int> &individual_limits);
    bool checkNrMatches(const std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> &matches_in);
    void filterKeypointsClassIDAffine(std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &kp_descr_in, 
                                      std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> &matches_in);
    void filterKeypointsClassIDAffineThreadFunc(const int startIdx, 
                                                const int endIdx, 
                                                const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &kp_descr_in, 
                                                const std::unordered_map<std::pair<int, int>, std::vector<cv::DMatch>, pair_hash, pair_EqualTo> &matches_in, 
                                                AffineMatchesFilterData &data);
    bool findAdditionalKeypointsLessTexture();
    void prepareImgsLessTextureThreadFunc(const int startIdx, 
                                          const int endIdx, 
                                          const std::vector<int> &camImgIdxsSearchAddKp, 
                                          std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap_boosted, 
                                          std::unordered_map<int, cv::Rect> &imgROIs);


    const double img_scaling;
    const std::vector<std::string> keypoint_types_;
    const std::string descriptor_type_;
    const std::string matcher_type_;
    bool affineInvariantUsed = false;
    const bool equalizeImgs_;
    const size_t nr_keypoint_types;
    int cpuCnt = -2;
    const int largest_cam_idx;
    const bool haveMasks;
    const int nrFeaturesToExtract;
    double kp_time_ms = -1., descr_time_ms = -1., match_time_ms = -1.;
    // camera index: rgb image, mask image
    std::unordered_map<int, std::pair<std::string, std::string>> img_mask_names;
    std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> imageMap;
    std::vector<int> indices;
    std::vector<std::pair<std::string, int>> indices_kp;
    // KEYPOINT_TYPES: camera index: vector of keypoints
    std::unordered_map<std::string, std::unordered_map<int, std::vector<cv::KeyPoint>>> init_keypoints;
    // camera index: vector of keypoints
    std::unordered_map<int, std::vector<cv::KeyPoint>> keypoints_combined;
    // camera index: all descriptors for every keypoint type
    std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> keypoints_descriptors;
    std::vector<std::pair<int, int>> cam_pair_idx;
    // first cam camera index, second cam camera index: all matches - filtered
    std::unordered_map<std::pair<int, int>, MatchData, pair_hash, pair_EqualTo> matches_filt;
    std::string imgs_ID;
};

} // namepace matchinglib
