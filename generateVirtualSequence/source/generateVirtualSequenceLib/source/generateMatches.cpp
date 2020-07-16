//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2020 Josef Maier
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

#include "generateMatches.h"
#include "io_data.h"
#include "imgFeatures.h"
#include <iomanip>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/eigen.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include "helper_funcs.h"
#include "side_funcs.h"
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>

using namespace std;
using namespace cv;

/* --------------------- Function prototypes --------------------- */

bool genParsStorePath(const std::string &basePath, const std::string &subpath, std::string &resPath);
bool checkOverwriteFiles(const std::string &filename, const std::string &errmsg, bool &overwrite);
bool checkOverwriteDelFiles(const std::string &filename, const std::string &errmsg, bool &overwrite);
template<typename T>
void shuffleVector(std::vector<T> &idxs, size_t si);
template<typename T, typename T1>
void reOrderVector(std::vector<T> &reOrderVec, std::vector<T1> &idxs);
void reOrderSortMatches(std::vector<cv::DMatch> &matches,
                        cv::Mat &descriptor1,
                        cv::Mat &descriptor2,
                        std::vector<cv::KeyPoint> &kp1,
                        std::vector<cv::KeyPoint> &kp2,
                        std::vector<cv::KeyPoint> &kp2NoError,
                        std::vector<bool> &inliers,
                        std::vector<cv::Mat> &homos,
                        std::vector<std::pair<std::pair<size_t,cv::KeyPoint>, std::pair<size_t,cv::KeyPoint>>> &srcImgIdxAndKp,
                        std::vector<int> &corrType,
                        std::vector<cv::Mat> &homosCam1,
                        std::vector<size_t> &idxs1,
                        std::vector<size_t> &idxs2);
bool getNrEntriesYAML(const std::string &filename, const string &buzzword, int &nrEntries);
bool getImgROIs(const cv::Mat &H,
                const cv::Point2i &midPt,
                const int &minSqrROIimg2,
                cv::Rect &patchROIimg1,
                cv::Rect &patchROIimg2,
                cv::Rect &patchROIimg21,
                bool &reflectionX,
                bool &reflectionY,
                const cv::Size &imgFeatureSi,
                const cv::KeyPoint &kp1,
                const int &maxPatchSizeMult2);
void getRotationStats(const std::vector<cv::Mat> &Rs,
                      qualityParm &stats_roll,
                      qualityParm &stats_pitch,
                      qualityParm &stats_yaw);
void getTranslationStats(const std::vector<cv::Mat> &ts,
                         qualityParm &stats_tx,
                         qualityParm &stats_ty,
                         qualityParm &stats_tz);
bool correctMatchIdx(const unordered_set<int> &idxs, const vector<int> &missingIdxs, int &queryTrainIdx);
void correctMatchesIdx(vector<DMatch> &matches,
                       bool isLeftKp,
                       const unordered_set<int> &idxs,
                       const vector<int> &missingIdxs,
                       vector<int> &delKpIdx);
void getNewFromOldIdx(const vector<int> &missingIdxs, int &oldIdx);
void showCorrespPatches(const cv::Mat &patch1, const cv::Mat &patch2);
void showMatchesGTMpart(const std::vector<cv::DMatch> &matchesGT,
                        const std::vector<bool> &leftInlier,
                        const std::vector<cv::KeyPoint> &keypL,
                        const std::vector<cv::KeyPoint> &keypR,
                        const cv::Mat &img1,
                        const cv::Mat &img2);
void reOrderDescriptors(cv::Mat &descriptors, const std::vector<int> &idxs);
bool checkMatchesInvalidValues(const std::vector<cv::DMatch> &matches,
                               const std::vector<cv::KeyPoint> &kp1,
                               const std::vector<cv::KeyPoint> &kp2,
                               const cv::Size &img1Size,
                               const cv::Size &img2Size);

/* -------------------------- Functions -------------------------- */

size_t TNTPindexer::getNextTPID(const size_t &tp_repPatt_idx, const size_t &min_frm_idx){
    if(tp_ids.empty()){
        throw SequenceException("No TP match IDs available");
    }
    if(tp_repPatt_idxs.find(tp_repPatt_idx) != tp_repPatt_idxs.end()){
        nrCorrs++;
        return tp_ids[tp_repPatt_idxs.at(tp_repPatt_idx).first];
    }else if(tn_repPatt_idxs.find(tp_repPatt_idx) != tn_repPatt_idxs.end()){
        if(!tp_repPatt_idxs.empty()) {
            size_t min_idx = tn_repPatt_idxs.at(tp_repPatt_idx).second;
            size_t min_idx2 = tp_repPatt_idxs.rbegin()->second.second;
            bool isStereoRep = min_idx == min_frm_idx;
            if (isStereoRep && (min_idx2 == min_frm_idx)) {
                std::vector<size_t> stereo_idxs;
                std::map<size_t, std::pair<size_t, size_t>>::reverse_iterator rit;
                for (rit = tp_repPatt_idxs.rbegin(); rit != tp_repPatt_idxs.rend(); ++rit) {
                    if (rit->second.second == min_frm_idx) {
                        stereo_idxs.push_back(rit->first);
                    } else {
                        break;
                    }
                }
                const size_t new_rep_idx = stereo_idxs[rand2() % stereo_idxs.size()];
                const std::pair<size_t, size_t> tmp = tp_repPatt_idxs.at(new_rep_idx);
                tp_repPatt_idxs.emplace(tp_repPatt_idx, tmp);
                nrCorrs++;
                return tp_ids[tmp.first];
            } else if (!isStereoRep) {
                std::vector<size_t> mono_idxs;
                std::map<size_t, std::pair<size_t, size_t>>::reverse_iterator rit;
                for (rit = tp_repPatt_idxs.rbegin(); rit != tp_repPatt_idxs.rend(); ++rit) {
                    if (rit->second.second == min_idx) {
                        mono_idxs.push_back(rit->first);
                    } else if (rit->second.second < min_idx) {
                        break;
                    }
                }
                if (!mono_idxs.empty()) {
                    const size_t new_rep_idx = mono_idxs[rand2() % mono_idxs.size()];
                    const std::pair<size_t, size_t> tmp = tp_repPatt_idxs.at(new_rep_idx);
                    tp_repPatt_idxs.emplace(tp_repPatt_idx, tmp);
                    nrCorrs++;
                    return tp_ids[tmp.first];
                }
            }
        }
    }
    size_t idx = tp_idx;
    size_t tp_repPatt_idx_ = tp_repPatt_idx;
    if(tp_idx >= tp_ids.size()){
        idx = rand2() % tp_ids.size();
        tp_repPatt_idx_ = tp_repPatt_idxs.rbegin()->first + 1;
    }else if(tp_repPatt_idx_ == numeric_limits<size_t>::max()){
        tp_repPatt_idx_ = tp_repPatt_idxs.rbegin()->first + 1;
    }
//    tptn_ids.emplace(tp_idx + tn_idx, &tp_ids[idx]);
    tp_repPatt_idxs.emplace(tp_repPatt_idx_, std::make_pair(idx, min_frm_idx));
    tp_idx++;
    nrCorrs++;
    return tp_ids[idx];
}

size_t TNTPindexer::getNextTNID(const size_t &tn_repPatt_idx, const size_t &min_frm_idx){
    if(tn_ids.empty()){
        throw SequenceException("No TN match IDs available");
    }
    if(tn_repPatt_idxs.find(tn_repPatt_idx) != tn_repPatt_idxs.end()){
        nrCorrs++;
        return tn_ids[tn_repPatt_idxs.at(tn_repPatt_idx).first];
    }else if(tp_repPatt_idxs.find(tn_repPatt_idx) != tp_repPatt_idxs.end()){
        if(!tn_repPatt_idxs.empty()) {
            size_t min_idx = tp_repPatt_idxs.at(tn_repPatt_idx).second;
            size_t min_idx2 = tn_repPatt_idxs.rbegin()->second.second;
            bool isStereoRep = min_idx == min_frm_idx;
            if (isStereoRep && (min_idx2 == min_frm_idx)) {
                std::vector<size_t> stereo_idxs;
                std::map<size_t, std::pair<size_t, size_t>>::reverse_iterator rit;
                for (rit = tn_repPatt_idxs.rbegin(); rit != tn_repPatt_idxs.rend(); ++rit) {
                    if (rit->second.second == min_frm_idx) {
                        stereo_idxs.push_back(rit->first);
                    } else {
                        break;
                    }
                }
                const size_t new_rep_idx = stereo_idxs[rand2() % stereo_idxs.size()];
                const std::pair<size_t, size_t> tmp = tn_repPatt_idxs.at(new_rep_idx);
                tn_repPatt_idxs.emplace(tn_repPatt_idx, tmp);
                nrCorrs++;
                return tn_ids[tmp.first];
            } else if (!isStereoRep) {
                std::vector<size_t> mono_idxs;
                std::map<size_t, std::pair<size_t, size_t>>::reverse_iterator rit;
                for (rit = tn_repPatt_idxs.rbegin(); rit != tn_repPatt_idxs.rend(); ++rit) {
                    if (rit->second.second == min_idx) {
                        mono_idxs.push_back(rit->first);
                    } else if (rit->second.second < min_idx) {
                        break;
                    }
                }
                if (!mono_idxs.empty()) {
                    const size_t new_rep_idx = mono_idxs[rand2() % mono_idxs.size()];
                    const std::pair<size_t, size_t> tmp = tn_repPatt_idxs.at(new_rep_idx);
                    tn_repPatt_idxs.emplace(tn_repPatt_idx, tmp);
                    nrCorrs++;
                    return tn_ids[tmp.first];
                }
            }
        }
    }
    size_t idx = tn_idx;
    size_t tn_repPatt_idx_ = tn_repPatt_idx;
    if(tn_idx >= tn_ids.size()){
        idx = rand2() % tn_ids.size();
//        tn_repPatt_idx_ = tn_repPatt_idxs.rbegin()->first + 1;
    }
    if(tn_repPatt_idx_ == numeric_limits<size_t>::max()){
        tn_repPatt_idx_ = tn_repPatt_idxs.rbegin()->first + 1;
    }
//    tptn_ids.emplace(tp_idx + tn_idx, &tn_ids[idx]);
    tn_repPatt_idxs.emplace(tn_repPatt_idx_, std::make_pair(idx, min_frm_idx));
    tn_idx++;
    nrCorrs++;
    return tn_ids[idx];
}

size_t TNTPindexer::getCorrID(const size_t &nr, const bool &isTN, const bool &tryBoth){
    std::map<size_t, std::pair<size_t, size_t>>::const_iterator gottntp;
    if(isTN){
        gottntp = tn_repPatt_idxs.find(nr);
        if(gottntp == tn_repPatt_idxs.end()){
            if(tryBoth){
                gottntp = tp_repPatt_idxs.find(nr);
                if(gottntp == tp_repPatt_idxs.end()){
                    throw SequenceException("Feature number out of range");
                }
                return tp_ids[gottntp->second.first];
            }
            throw SequenceException("Feature number out of range");
        }
        return tn_ids[gottntp->second.first];
    }
    gottntp = tp_repPatt_idxs.find(nr);
    if(gottntp == tp_repPatt_idxs.end()){
        if(tryBoth){
            gottntp = tn_repPatt_idxs.find(nr);
            if(gottntp == tn_repPatt_idxs.end()){
                throw SequenceException("Feature number out of range");
            }
            return tn_ids[gottntp->second.first];
        }
        throw SequenceException("Feature number out of range");
    }
    return tp_ids[gottntp->second.first];
}

genMatchSequ::genMatchSequ(const std::string &sequLoadFolder,
                           GenMatchSequParameters &parsMtch_,
                           uint32_t verbose_,
                           const std::string &writeIntermRes_path_) :
        genStereoSequ(false, verbose_, writeIntermRes_path_),
        parsMtch(parsMtch_),
        sequParsLoaded(true),
        tntpindexer(&rand2){
    CV_Assert(parsMtch.parsValid);

    sequLoadPath = sequLoadFolder;
    if(sequLoadPath.empty()){
        if(parsMtch.mainStorePath.empty()){
            throw SequenceException("No path for loading 3D sequence provided!");
        }
        else{
            sequLoadPath = parsMtch.mainStorePath;
        }
    }
    else if(parsMtch.mainStorePath.empty()){
        parsMtch.mainStorePath = sequLoadPath;
    }

    if (!checkPathExists(sequLoadPath)) {
        string errmsg = "Path for loading 3D sequence does not exist: " + sequLoadPath;
        throw SequenceException(errmsg);
    }
    genSequenceParsFileName();

    string filename = concatPath(sequLoadPath, sequParFileName);
    if(!checkFileExists(filename)){
        string errmsg = "Necessary 3D sequence file " +
                        filename + " does not exist!";
        throw SequenceException(errmsg);
    }

    if (!readSequenceParameters(filename)) {
        string errmsg = "Necessary 3D sequence file " +
                        filename + " could not be loaded!";
        throw SequenceException(errmsg);
    }

    if(!readPointClouds(sequLoadPath, pclBaseFName)){
        string errmsg = "Necessary 3D PCL point cloud files could not be loaded!";
        throw SequenceException(errmsg);
    }

    long int seed = randSeed(rand_gen);
    randSeed(rand2, seed);

    K1i = K1.inv();
    K2i = K2.inv();
    setCamMats(K1, K2);
    kpErrors.clear();
    featureIdxBegin = 0;
    adaptPatchSize();
}

void genMatchSequ::genSequenceParsFileName() {
    const std::string sequParFileNameBase = "sequPars";

    sequParFileName = genSequFileExtension(sequParFileNameBase);
}

std::string genMatchSequ::genSequFileExtension(const std::string &basename) const{
    std::string filename = basename;

    if (parsMtch.rwXMLinfo) {
        filename += ".xml";
    } else {
        filename += ".yaml";
    }

    if (parsMtch.compressedWrittenInfo) {
        filename += ".gz";
    }

    return filename;
}

bool genMatchSequ::genSequenceParsStorePath(){
    if(!sequParsLoaded || (sequLoadPath != parsMtch.mainStorePath)) {
        if(sequParsLoaded){
            pars = pars3D;
        }
        hash_Sequ = hashFromSequPars();
        bool sequPathExists = false;
        if(sequParsLoaded){
            if(!checkPathExists(parsMtch.mainStorePath)){
                cerr << "Given path " << parsMtch.mainStorePath << " to store results does not exist!" << endl;
                return false;
            }
            string resPath = concatPath(parsMtch.mainStorePath, std::to_string(hash_Sequ));
            if(checkPathExists(resPath)){
                sequPathExists = true;
                sequParPath = resPath;
            }
        }
        if(!sequPathExists) {
            if (!genParsStorePath(parsMtch.mainStorePath, std::to_string(hash_Sequ), sequParPath)) {
                return false;
            }
        }
    }else{
        sequParPath = sequLoadPath;
    }
    return true;
}

bool genParsStorePath(const std::string &basePath, const std::string &subpath, std::string &resPath){
    if(!checkPathExists(basePath)){
        cerr << "Given path " << basePath << " to store results does not exist!" << endl;
        return false;
    }

    resPath = concatPath(basePath, subpath);
    int idx = 0;
    while(checkPathExists(resPath)){
        resPath = concatPath(basePath, subpath + "_" + std::to_string(idx));
        idx++;
        if(idx > 10000){
            cerr << "Cannot create a path for storing results as more than 10000 variants of the same path exist." << endl;
            return false;
        }
    }
    if(!createDirectory(resPath)){
        cerr << "Unable to create directory for storing results: " << resPath << endl;
        return false;
    }

    return true;
}

bool genMatchSequ::genMatchDataStorePath(){
    hash_Matches = hashFromMtchPars();
    return genParsStorePath(sequParPath, std::to_string(hash_Matches), matchDataPath);
}

bool genMatchSequ::generateMatches(){
    //Calculate maximum number of TP and TN correspondences
    totalNrCorrs();

    //Calculate GTM
    if(!calcGTM() && !onlyWarped && (verbose & PRINT_WARNING_MESSAGES)){
        cout << "Use of GTM disabled." << endl;
    }

    //Calculate features from given images
    if(!getFeatures()){
        cerr << "Unable to calculate necessary keypoints from images!" << endl;
        return false;
    }

    //Generate path to store results
    if(!genSequenceParsStorePath()){
        return false;
    }
    if(!genMatchDataStorePath()){
        return false;
    }
    if(!writeMatchingParameters()){
        cerr << "Unable to store matching parameters!" << endl;
        return false;
    }

    if(!sequParsLoaded || (sequLoadPath != parsMtch.mainStorePath)) {
        if(sequParsLoaded){
            string filename;
            if(!getSequenceOverviewParsFileName(filename)){
                return false;
            }
            if(!checkFileExists(filename)) {
                //pars = pars3D;
                if(!writeSequenceOverviewPars()){
                    cerr << "Unable to write file with sequence parameter overview!" << endl;
                    return false;
                }
            }
        }else {
            if (!writeSequenceOverviewPars()) {
                cerr << "Unable to write file with sequence parameter overview!" << endl;
                return false;
            }
        }
    }

    chrono::high_resolution_clock::time_point t1, t2, t3;

    bool overwriteFiles = false;
    bool overWriteSuccess = true;
    while (actFrameCnt < nrFramesGenMatches){
        t1 = chrono::high_resolution_clock::now();
        if(!sequParsLoaded) {
            //Generate 3D correspondences for a single frame
            startCalc_internal();
            actFrameCnt--;
            //Write the result to disk
            if(parsMtch.storePtClouds && overWriteSuccess){
                string singleFrameDataFName = sequSingleFrameBaseFName + "_" + std::to_string(actFrameCnt);
                singleFrameDataFName = genSequFileExtension(singleFrameDataFName);
                singleFrameDataFName = concatPath(sequParPath, singleFrameDataFName);
                if(checkOverwriteDelFiles(singleFrameDataFName,
                                          "Output file for sequence parameters already exists:", overwriteFiles)){
                    write3DInfoSingleFrame(singleFrameDataFName);
                }
                else{
                    overWriteSuccess = false;
                }
            }
        }else{
            //Load sequence data for the actual frame
            string singleFrameDataFName = sequSingleFrameBaseFName + "_" + std::to_string(actFrameCnt);
            singleFrameDataFName = genSequFileExtension(singleFrameDataFName);
            singleFrameDataFName = concatPath(sequLoadPath, singleFrameDataFName);
            if(!checkFileExists(singleFrameDataFName)){
                cerr << "3D correspondence file for single frame does not exist: " <<
                singleFrameDataFName << endl;
                return false;
            }
            if(!read3DInfoSingleFrame(singleFrameDataFName)){
                cerr << "Unable to load 3D correspondence file for single frame: " <<
                     singleFrameDataFName << endl;
                return false;
            }
        }
        t2 = chrono::high_resolution_clock::now();

        //Calculate matches for actual frame
        actTransGlobWorld = Mat::eye(4,4,CV_64FC1);
        absCamCoordinates[actFrameCnt].R.copyTo(actTransGlobWorld.rowRange(0,3).colRange(0,3));
        absCamCoordinates[actFrameCnt].t.copyTo(actTransGlobWorld.col(3).rowRange(0,3));
        actTransGlobWorldit = actTransGlobWorld.inv().t();//Will be needed to transfer a plane from the local to the world coordinate system

        //Calculate the norm of the translation vector of the actual stereo configuration
        actNormT = norm(actT);

        //Update linear correspondence index
        updateLinearIdx();

        //Get most used images of actual frame
        getMostUsedImgs();

        //Generate matching keypoints and descriptors and store them (in addition to other information) to disk
        if(!generateCorrespondingFeatures()){
            return false;
        }

//        if(sequParsLoaded){
            actFrameCnt++;
//        }

        //Calculate execution time
        t3 = chrono::high_resolution_clock::now();
        timePerFrameMatch.emplace_back(make_pair(chrono::duration_cast<chrono::microseconds>(t2 - t1).count(),
                                                 chrono::duration_cast<chrono::microseconds>(t3 - t2).count()));
    }


    if(!sequParsLoaded && parsMtch.storePtClouds && overWriteSuccess){
        //Calculate statistics on the execution time for 3D scenes
        getStatisticfromVec(timePerFrame, time3DStats);

        string sequParFName = concatPath(sequParPath, sequParFileName);
        if(checkOverwriteDelFiles(sequParFName,
                "Output file for sequence parameters already exists:", overwriteFiles)) {
            if (!writeSequenceParameters(sequParFName)){
                cerr << "Unable to write sequence parameters to disk: " << sequParFName << endl;
                overWriteSuccess = false;
            }
        }else{
            overWriteSuccess = false;
        }
        if(overWriteSuccess) {
            if(!writePointClouds(sequParPath, pclBaseFName, overwriteFiles)){
                cerr << "Unable to write PCL point clouds to disk!" << endl;
            }
        }
    }

    //Calculate statistics for the execution time of generating matches
    vector<double> matchTime;
    matchTime.reserve(timePerFrameMatch.size());
    for(auto &i : timePerFrameMatch){
        matchTime.push_back(i.second);
    }
    getStatisticfromVec(matchTime, timeMatchStats);

    //Calculate mean and standard deviation for keypoint accuracies
    //Calculate the mean
    double mean_kp_error = 0;
    for(auto& i : kpErrors){
        mean_kp_error += i;
    }
    mean_kp_error /= (double)kpErrors.size();

    //Calculate the standard deviation
    double std_dev_kp_error = 0;
    for(auto& i : kpErrors){
        const double diff = i - mean_kp_error;
        std_dev_kp_error += diff * diff;
    }
    std_dev_kp_error /= (double)kpErrors.size();
    std_dev_kp_error = sqrt(std_dev_kp_error);

    //Write the error statistic, image names, and execution times to a separate file
    writeKeyPointErrorAndSrcImgs(mean_kp_error, std_dev_kp_error);

    return true;
}



void genMatchSequ::addImgNoiseGauss(const cv::Mat &patchIn,
        cv::Mat &patchOut,
        double meanNoise,
        double stdNoise,
        bool visualize){
    Mat mSrc_16SC;
    Mat mGaussian_noise = Mat(patchIn.size(),CV_16SC1);
    randn(mGaussian_noise,Scalar::all(meanNoise), Scalar::all(stdNoise));

    patchIn.convertTo(mSrc_16SC,CV_16SC1);
    addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
    mSrc_16SC.convertTo(patchOut,patchIn.type());

    if(visualize){
        Mat bothpatches;
        hconcat(patchIn,patchOut,bothpatches);
        namedWindow("Feature patch with and without gaussian noise", WINDOW_AUTOSIZE);
        imshow("Feature patch with and without gaussian noise", bothpatches);

        waitKey(0);
        destroyWindow("Feature patch with and without gaussian noise");
    }
}

void genMatchSequ::addImgNoiseSaltAndPepper(const cv::Mat &patchIn,
        cv::Mat &patchOut,
        int minTH,
        int maxTH,
        bool visualize){
    Mat saltpepper_noise = Mat::zeros(patchIn.size(),CV_8UC1);
    randu(saltpepper_noise,0,255);

    Mat black = saltpepper_noise < minTH;
    Mat white = saltpepper_noise > maxTH;

    patchOut = patchIn.clone();
    patchOut.setTo(255,white);
    patchOut.setTo(0,black);

    if(visualize){
        Mat bothpatches;
        hconcat(patchIn,patchOut,bothpatches);
        namedWindow("Feature patch with and without salt and pepper noise", WINDOW_AUTOSIZE);
        imshow("Feature patch with and without salt and pepper noise", bothpatches);

        waitKey(0);
        destroyWindow("Feature patch with and without salt and pepper noise");
    }
}

//Create a homography for a TN correspondence
cv::Mat genMatchSequ::getHomographyForDistortionTN(const cv::Mat& x1,
                                     bool visualize){
    //Calculate a 3D point
    Mat b1 = K1i * x1;
    double zacc = std::log10(actNormT / 100.0);
    if(zacc > 0){
        zacc = 1;
    }else{
        zacc = std::pow(10.0, ceil(abs(zacc)));
    }
    //Take random depth range
    size_t depthRegion = rand2() % 3;
    double z_min, z_max;
    switch(depthRegion){
        case 0:
            z_min = actDepthNear;
            z_max = actDepthMid;
            break;
        case 1:
            z_min = actDepthMid;
            z_max = actDepthFar;
            break;
        case 2:
            z_min = actDepthFar;
            z_max = maxFarDistMultiplier * actDepthFar;
            break;
        default:
            z_min = actDepthMid;
            z_max = actDepthFar;
            break;
    }
    double z = round(zacc * getRandDoubleValRng(z_min, z_max)) / zacc;
    double s1 = z / b1.at<double>(2);
    Mat X = s1 * b1;
    Mat x2 = K2 * (actR * X + actT);
    x2 /= x2.at<double>(2);

    return getHomographyForDistortion(X, x1, x2, -1, 0, cv::noArray(), visualize);
}

//Check, if used 3D point in the camera coordinate system is the same as pointed by the index (in world coordinates)
bool genMatchSequ::check3DToIdxConsisty(const cv::Mat &X, const int64_t &idx3D, const int64_t &idx3D2){
    if(!planeTo3DIdx.empty() && (planeTo3DIdx.find(idx3D) != planeTo3DIdx.end())){
        cv::Mat X1 = absCamCoordinates[actFrameCnt].R * X + absCamCoordinates[actFrameCnt].t;
        cv::Mat X2;
        if (idx3D < 0) {
            auto frNr_init = static_cast<size_t>((-1 * idx3D) & 0xFFFFFFFF) >> 8;
            auto idx = static_cast<size_t>(((-1 * idx3D) >> 32) - 1);
            auto mvObjNr_init = static_cast<size_t>((-1 * idx3D) & 0xFF) - 1;
            unsigned int sum_of_elems = 0;
            for (size_t i = 0; i < frNr_init; ++i) {
                sum_of_elems += nrMovingObjPerFrame[i];
            }
            sum_of_elems += mvObjNr_init;
            pcl::PointXYZ X_tmp = movObj3DPtsWorldAllFrames[sum_of_elems][idx];
            X2 = (Mat_<double>(3, 1) << (double) X_tmp.x, (double) X_tmp.y, (double) X_tmp.z);
            auto frNr1 = get<2>(planeTo3DIdx[idx3D]);
            auto frNr_diff = actFrameCnt - frNr1;
            Mat tdiff = Mat::zeros(3, 1, CV_64FC1);
            if (frNr_diff > 0) {
                auto mvObjNr_act = static_cast<size_t>((-1 * idx3D2) & 0xFFFFFFFF) - 1;
                get<2>(movObjFrameEmerge[mvObjNr_act]).copyTo(tdiff);
                tdiff *= static_cast<double>(frNr_diff);
            }
            X1 -= tdiff;
        }else{
            auto idx = static_cast<size_t>(idx3D);
            pcl::PointXYZ X_tmp = staticWorld3DPts->at(idx);
            X2 = (Mat_<double>(3, 1) << (double) X_tmp.x, (double) X_tmp.y, (double) X_tmp.z);
        }
        Mat diff = X2 - X1;
        double dn = cv::norm(diff);
        bool test = nearZero(dn / 750.0);
        return test;
    }else{
        return true;
    }
}

//Check, if 2D correspondences of both stereo cameras are projections of corresponding 3D point and provided 3D point and world 3D point pointed to by index are the same
bool genMatchSequ::check2D3DConsistency(const cv::Mat &x1, const cv::Mat &x2, const cv::Mat &X, const int64_t &idx3D, const int64_t &idx3D2){
    bool check1 = checkCorrespondenceConsisty(x1, x2, X);
    bool check2 = check3DToIdxConsisty(X, idx3D, idx3D2);
    return check1 && check2;
}

//Checks, if a plane was already calculated for the given 3D coordinate and if yes, adapts the plane.
//If not, a new plane is calculated.
cv::Mat genMatchSequ::getHomographyForDistortionChkOld(const cv::Mat& X,
                                                       const cv::Mat& x1,
                                                       const cv::Mat& x2,
                                                       int64_t idx3D,
                                                       int64_t idx3D2,
                                                       size_t keyPIdx,
                                                       bool visualize,
                                                       bool forCam1){
    if(!forCam1) {
        if (!check2D3DConsistency(x1, x2, X, idx3D, idx3D2)) {
            throw SequenceException("2D is not consistend with 3D for calculating homographies!");
        }
    }
    if(!planeTo3DIdx.empty() && (idx3D != -1)){
        if ((planeTo3DIdx.find(idx3D) != planeTo3DIdx.end()) && (get<2>(planeTo3DIdx[idx3D]) != actFrameCnt)) {
            Mat trans = Mat::eye(4, 4, CV_64FC1);
            trans.rowRange(0, 3).colRange(0, 3) = absCamCoordinates[actFrameCnt].R.t();
            trans.col(3).rowRange(0, 3) =
                    -1.0 * absCamCoordinates[actFrameCnt].R.t() * absCamCoordinates[actFrameCnt].t;
            Mat plane;
            if (idx3D >= 0) {
                trans = trans.inv().t();
                plane = trans * get<0>(planeTo3DIdx[idx3D]);
            } else{
//                auto frNr_init = static_cast<size_t>((-1 * idx3D) & 0xFFFFFFFF) >> 8;
//                auto frNr_diff = actFrameCnt - frNr_init;
//                auto mvObjNr_init = static_cast<size_t>((-1 * idx3D) & 0xFF) - 1;
//                unsigned int sum_of_elems = 0;
//                for(size_t i = 0; i < frNr_init; ++i){
//                    sum_of_elems += nrMovingObjPerFrame[i];
//                }
//                sum_of_elems += mvObjNr_init;
                auto frNr1 = get<2>(planeTo3DIdx[idx3D]);
                auto frNr_diff = actFrameCnt - frNr1;
                if (frNr_diff > 0) {
                    auto mvObjNr_act = static_cast<size_t>((-1 * idx3D2) & 0xFFFFFFFF) - 1;
                    Mat tdiff = Mat::zeros(3,1,CV_64FC1);
                    get<2>(movObjFrameEmerge[mvObjNr_act]).copyTo(tdiff);
                    tdiff *= static_cast<double>(frNr_diff);
                    tdiff = trans.rowRange(0, 3).colRange(0, 3) * tdiff;
                    trans.col(3).rowRange(0, 3) += tdiff;
//                    Mat trans2 = Mat::eye(4, 4, CV_64FC1);
//                    tdiff.copyTo(trans2.col(3).rowRange(0, 3));
//                    trans = trans * trans2;
                }
                trans = trans.inv().t();
                plane = trans * get<0>(planeTo3DIdx[idx3D]);
            }
            try {
                return getHomographyForDistortion(X, x1, x2, idx3D, keyPIdx, plane, visualize, forCam1);
            } catch (SequenceException &e) {
                cout << "Exception while recalculating old homography: " << e.what() << endl;
                throw;
            }
        }
    }
    try {
        return getHomographyForDistortion(X, x1, x2, idx3D, keyPIdx, cv::noArray(), visualize, forCam1);
    }catch(SequenceException &e){
        cout << "Exception while calculating new homography: " << e.what() << endl;
        throw;
    }
}

/*Calculates a homography by rotating a plane in 3D (which was generated using a 3D point and its projections into
 * camera 1 & 2) and backprojection of corresponding points on that plane into the second image
 *
 * X ... 3D cooridinate
 * x1 ... projection in cam1
 * x2 ... projection in cam2
 * planeNVec ... Plane parameters for an already calculated plane
 */
cv::Mat genMatchSequ::getHomographyForDistortion(const cv::Mat& X,
                                                 const cv::Mat& x1,
                                                 const cv::Mat& x2,
                                                 int64_t idx3D,
                                                 size_t keyPIdx,
                                                 cv::InputArray planeNVec,
                                                 bool visualize,
                                                 bool forCam1){
    CV_Assert((X.rows == 3) && (X.cols == 1));
    CV_Assert((x1.rows == 3) && (x1.cols == 1));
    CV_Assert((x2.rows == 3) && (x2.cols == 1));

    Mat pn, bn, p1;
    double d;
    Mat Xl1;
    if(planeNVec.empty()) {
        if(forCam1){
            return cv::Mat::eye(3, 3, CV_64FC1);
        }
        //Get the ray to X from cam 1
        Mat b1 = K1i * x1;
        //Get the ray direction to X from cam 2
        Mat b2 = actR.t() * K2i * x2;

        //Calculate the normal vector of the plane by taking the mean vector of both camera rays
        b1 /= norm(b1);
        b2 /= norm(b2);
        bn = (b1 + b2) / 2.0;
        bn /= norm(bn);

        //Get the line on the plane normal to both viewing rays
        Mat bpa = b1.cross(b2);
        bpa /= norm(bpa);

        //Get the normal to bn and bpa
        Mat bpb = bpa.cross(bn);
        bpb /= norm(bpb);

        //Get the rotation angles
        const double maxRotAngleAlpha = 5.0 * (M_PI - acos(b1.dot(b2) / (norm(b1) * norm(b2)))) / 16.0;
        const double maxRotAngleBeta = 5.0 * M_PI / 16.0;
        /*const double maxRotAngleAlpha = (M_PI - acos(b1.dot(b2) / (norm(b1) * norm(b2)))) / 4.0;
        const double maxRotAngleBeta = M_PI / 4.0;*/
        //alpha ... Rotation angle about first vector 'bpa' in the plane which is perpendicular to the plane normal 'bn'
        double alpha = getRandDoubleValRng(-maxRotAngleAlpha, maxRotAngleAlpha);
        //beta ... Rotation angle about second vector 'bpb' in the plane which is perpendicular to the plane normal 'bn' and 'bpa'
        double beta = getRandDoubleValRng(-maxRotAngleBeta, maxRotAngleBeta);

        //Rotate bpb about line bpa using alpha
        Mat bpbr = rotateAboutLine(bpa, alpha, bpb);

        //Rotate bpa about line bpb using beta
        Mat bpar = rotateAboutLine(bpb, beta, bpa);

        //Construct a plane from bpar and bpbr
        pn = bpbr.cross(bpar);//Normal of the plane
        pn /= norm(pn);
        //Plane parameters
        d = pn.dot(X);
        p1 = pn.clone();
        p1.push_back(-1.0 * d);

        /*Mat Xc = X.clone();
        Xc.push_back(1.0);
        double errVal = p1.dot(Xc);
        CV_Assert(nearZero(errVal));*/
        /*Transform the plane into the world coordinate system
         * actTransGlobWorld T = [R,t;[0,0,1]]
         * actTransGlobWorldit = (T^-1)^t
         * A point p is on the plane q if p.q=0
         * p' = T p
         * q' = (T^-1)^t q
         * Point p' is on plane q' when:  p'.q'=0
         * Then: p'.q' = p^t T^t (T^-1)^t q = p^t q = p.q
        */
        if(((idx3D >= 0) || !combCorrsImg12TP_IdxWorld2.empty()) && (idx3D != -1)) {
            if (!planeTo3DIdx.empty() && (planeTo3DIdx.find(idx3D) != planeTo3DIdx.end())) {
                throw SequenceException("Used plane to 3D index already exists!");
            }
            planeTo3DIdx[idx3D] = make_tuple(actTransGlobWorldit * p1, keyPIdx, actFrameCnt);
        }
    }else{
        p1 = planeNVec.getMat();
        CV_Assert((p1.rows == 4) && (p1.cols == 1));
        //Check if the plane is valid
        Mat Xh = X.clone();
        Xh.push_back(1.0);

        /*pcl::PointXYZ ptw = staticWorld3DPts->at(idx3D);
        Mat ptwm = (Mat_<double>(3,1) << (double)ptw.x, (double)ptw.y, (double)ptw.z);
        Mat c2wRnow = absCamCoordinates[actFrameCnt].R.clone();
        Mat c2wtnow = absCamCoordinates[actFrameCnt].t.clone();
        Mat c2wRlast = absCamCoordinates[actFrameCnt-1].R.clone();
        Mat c2wtlast = absCamCoordinates[actFrameCnt-1].t.clone();
        Mat ptwnow = c2wRnow.t() * ptwm - c2wRnow.t() * c2wtnow;
        Mat ptwlast = c2wRlast.t() * ptwm - c2wRlast.t() * c2wtlast;

        Mat diffM = ptwnow - X;
        double diff3DPos = cv::sum(diffM)[0];
        CV_Assert(nearZero(diff3DPos));

        Mat ptwlastc = ptwlast.clone();
        ptwlastc.push_back(1.0);

        Mat planeOld = get<0>(planeTo3DIdx[idx3D]).clone();
        Mat Pold = Mat::eye(4,4,CV_64FC1);
        Mat c2wRlasti = c2wRlast.t();
        Mat c2wtlasti = -1.0 * c2wRlasti * c2wtlast;
        c2wRlasti.copyTo(Pold.rowRange(0,3).colRange(0,3));
        c2wtlasti.copyTo(Pold.col(3).rowRange(0,3));
        Mat trans_old = Pold.inv().t();
        planeOld = trans_old * planeOld;

        double errValOld = planeOld.dot(ptwlastc);
        CV_Assert(nearZero(errValOld));*/

        double errorVal = p1.dot(Xh);
        if(!nearZero(errorVal / 750.0)){
            throw SequenceException("Used plane is not compatible with 3D coordinate!");
        }
        pn = p1.rowRange(0, 3).clone();
        d = -1.0 * p1.at<double>(3);
    }

    //Generate 3 additional image points in cam 1
    //Calculate a distance to the given projection for the additional points based on the distance and
    // account for numerical accuracy
    double dacc = max(5.0, 1e-3 * X.at<double>(2));
    Mat x12 = x1 + (Mat_<double>(3,1) << dacc, 0, 0);
    Mat x13 = x1 + (Mat_<double>(3,1) << 0, dacc, 0);
    Mat x14 = x1 + (Mat_<double>(3,1) << dacc, dacc, 0);

    //Generate lines from image points and their intersection with the plane
    Mat b12 = K1i * x12;
    b12 /= norm(b12);
    Mat b13 = K1i * x13;
    b13 /= norm(b13);
    Mat b14 = K1i * x14;
    b14 /= norm(b14);
    double t0 = d / pn.dot(b12);
    Mat X2 = t0 * b12;
    t0 = d / pn.dot(b13);
    Mat X3 = t0 * b13;
    t0 = d / pn.dot(b14);
    Mat X4 = t0 * b14;

    if(visualize && planeNVec.empty()) {
        //First plane parameters
        Mat p0 = bn.clone();
        double d0 = p0.dot(X);
        p0.push_back(-1.0 * d0);
        vector<Mat> pts3D;
        pts3D.push_back(X);
        pts3D.push_back(X2);
        pts3D.push_back(X3);
        pts3D.push_back(X4);
        visualizePlanes(pts3D, p0, p1);
    }

    //Project the 3 3D points into the second image
    Mat x21, x22, x23, x24;
    if(forCam1 && !planeNVec.empty()){
        Mat t21 = absCamCoordinates[actFrameCnt - 1].R.t() * (absCamCoordinates[actFrameCnt].t - absCamCoordinates[actFrameCnt - 1].t);
        Mat R21 = absCamCoordinates[actFrameCnt - 1].R.t() * absCamCoordinates[actFrameCnt].R;
        Mat X1 = R21 * X + t21;
        X2 = R21 * X2 + t21;
        X3 = R21 * X3 + t21;
        X4 = R21 * X4 + t21;
        x21 = K1 * X1;
        x21 /= x21.at<double>(2);
        x22 = K1 * X2;
        x23 = K1 * X3;
        x24 = K1 * X4;
    }else{
        x21 = x2;
        x22 = K2 * (actR * X2 + actT);
        x23 = K2 * (actR * X3 + actT);
        x24 = K2 * (actR * X4 + actT);
    }
    x22 /= x22.at<double>(2);
    x23 /= x23.at<double>(2);
    x24 /= x24.at<double>(2);

    //Calculate projective/perspective homography
    Mat x1all = Mat::ones(4,2, CV_64FC1);
    x1all.row(0) = x1.rowRange(0,2).t();
    x1all.row(1) = x12.rowRange(0,2).t();
    x1all.row(2) = x13.rowRange(0,2).t();
    x1all.row(3) = x14.rowRange(0,2).t();
    x1all.convertTo(x1all, CV_32FC1);
    Mat x2all = Mat::ones(4,2, CV_64FC1);
    x2all.row(0) = x21.rowRange(0,2).t();
    x2all.row(1) = x22.rowRange(0,2).t();
    x2all.row(2) = x23.rowRange(0,2).t();
    x2all.row(3) = x24.rowRange(0,2).t();
    x2all.convertTo(x2all, CV_32FC1);
    Mat H = getPerspectiveTransform(x1all, x2all);
    //Eliminate the translation
    Mat tm = H * x1;
    tm /= tm.at<double>(2);
    tm = x1 - tm;
    Mat tback = Mat::eye(3,3,CV_64FC1);
    tback.at<double>(0,2) = tm.at<double>(0);
    tback.at<double>(1,2) = tm.at<double>(1);
    H = tback * H;
    if(forCam1 && !planeNVec.empty()){
        H = H.inv();
    }
    return H.clone();
}

void genMatchSequ::visualizePlanes(std::vector<cv::Mat> &pts3D,
        const cv::Mat& plane1,
        const cv::Mat& plane2){

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("Plane for calculating homography"));

    Eigen::Affine3f m = initPCLViewerCoordinateSystems(viewer, Mat::eye(3,3,CV_64FC1),
                                                       Mat::zeros(3,1,CV_64FC1));
    Mat RC2 = actR.t();
    Mat tC2 = -1.0 * RC2 * actT;
    addVisualizeCamCenter(viewer, RC2, tC2);

    pcl::PointXYZ p1(0, 0, 0);
    pcl::PointXYZ p3((float)tC2.at<double>(0), (float)tC2.at<double>(1), (float)tC2.at<double>(2));
    for(size_t i = 0; i < pts3D.size(); ++i) {
        pcl::PointXYZ p2((float)pts3D[i].at<double>(0), (float)pts3D[i].at<double>(1), (float)pts3D[i].at<double>(2));
        viewer->addLine(p1, p2, "LineCam1_" + std::to_string(i));
        viewer->addLine(p3, p2, "LineCam2_" + std::to_string(i));
    }

    pcl::ModelCoefficients plane_coeff;
    plane_coeff.values.resize (4);
    plane_coeff.values[0] = (float)plane1.at<double>(0);
    plane_coeff.values[1] = (float)plane1.at<double>(1);
    plane_coeff.values[2] = (float)plane1.at<double>(2);
    plane_coeff.values[3] = (float)plane1.at<double>(3);
    viewer->addPlane(plane_coeff,
                     (float)pts3D[0].at<double>(0),
                     (float)pts3D[0].at<double>(1),
                     (float)pts3D[0].at<double>(2),
                     "initPlane");
    /*viewer->addPlane(plane_coeff,
                     "initPlane");*/
    plane_coeff.values[0] = (float)plane2.at<double>(0);
    plane_coeff.values[1] = (float)plane2.at<double>(1);
    plane_coeff.values[2] = (float)plane2.at<double>(2);
    plane_coeff.values[3] = (float)plane2.at<double>(3);
    viewer->addPlane(plane_coeff,
                     (float)pts3D[0].at<double>(0),
                     (float)pts3D[0].at<double>(1),
                     (float)pts3D[0].at<double>(2),
                     "resPlane");
    /*viewer->addPlane(plane_coeff,
                     "resPlane");*/

    setPCLViewerCamPars(viewer, m.matrix(), K1);

    startPCLViewer(viewer);
}

//Rotates a line 'b' about a line 'a' (only direction vector) using the given angle
cv::Mat genMatchSequ::rotateAboutLine(const cv::Mat &a, const double &angle, const cv::Mat &b){
    CV_Assert((a.rows == 3) && (a.cols == 1));
    CV_Assert((b.rows == 3) && (b.cols == 1));
    CV_Assert((angle < M_PI) && (angle > -M_PI));

    Mat a_ = a.clone();
    a_ /= norm(a_);

    double checkSum = cv::sum(a_)[0];
    if(nearZero(checkSum) || nearZero(angle)){
        return b.clone();
    }

    //Check if the rotation axis is identical to the x, y, or z-axis
    if(nearZero(a_.at<double>(1)) && nearZero(a_.at<double>(2))){
        Mat R_x = (Mat_<double>(3,3) <<
                1.0, 0, 0,
                0, cos(angle), -sin(angle),
                0, sin(angle), cos(angle));
        return R_x * b;
    }
    else if(nearZero(a_.at<double>(0)) && nearZero(a_.at<double>(2))){
        Mat R_y = (Mat_<double>(3,3) <<
                cos(angle), 0, sin(angle),
                0, 1.0, 0,
                -sin(angle), 0, cos(angle));
        return R_y * b;
    }
    else if(nearZero(a_.at<double>(0)) && nearZero(a_.at<double>(1))){
        Mat R_z = (Mat_<double>(3,3) <<
                cos(angle), -sin(angle), 0,
                sin(angle), cos(angle), 0,
                0, 0, 1.0);
        return R_z * b;
    }

    const double c = sqrt(a_.at<double>(1) * a_.at<double>(1) + a_.at<double>(2) * a_.at<double>(2));
    const double sinx = a_.at<double>(1) / c;
    const double cosx = a_.at<double>(2) / c;
    Mat R_x = (Mat_<double>(3,3) <<
            1.0, 0, 0,
            0, cosx, -sinx,
            0, sinx, cosx);
    auto &siny = a_.at<double>(0);
    const double &cosy = c;
    Mat R_y = (Mat_<double>(3,3) <<
            cosy, 0, siny,
            0, 1.0, 0,
            -siny, 0, cosy);
    Mat R_z = (Mat_<double>(3,3) <<
            cos(angle), -sin(angle), 0,
            sin(angle), cos(angle), 0,
            0, 0, 1.0);
    return R_x.t() * R_y.t() * R_z * R_y * R_x * b;
}

//Load the images in the given folder with a given image pre- and/or postfix (supports wildcards)
bool genMatchSequ::getImageList() {
    std::vector<std::string> filenames;
    bool err = loadImageSequenceNew(parsMtch.imgPath,
                                    parsMtch.imgPrePostFix, imageList);
    bool err1 = getImageNetImgs(filenames);
    if(err1){
        imageList.insert(imageList.end(), filenames.begin(), filenames.end());
    }
    return err || err1;
}

//Download and load image names from ImageNet
bool genMatchSequ::getImageNetImgs(std::vector<std::string> &filenames){
    using namespace boost::interprocess;
    if(parsMtch.nrImgsFromImageNet <= 0){
        return false;
    }
    if(parsMtch.execPath.empty()){
        cout << "Please provide path of this executable to use images from ImageNet." << endl;
        return false;
    }
    string par_path = getParentPath(parsMtch.execPath);
    string py_file = concatPath(par_path, "downloadImageNet.py");
    if(!checkFileExists(py_file)){
        cerr << "Cannot find Python file " << py_file << endl;
        return false;
    }
    string imgNetP = concatPath(parsMtch.imgPath, "ImageNet");
    if(!checkPathExists(imgNetP)){
        if(!createDirectory(imgNetP)){
            cerr << "Cannot create directory " << imgNetP << endl;
            return false;
        }
    }
    string command = "python3 " + py_file;
    command += " --path_store ";
    command += imgNetP;
    if(!parsMtch.imageNetIDs.empty()){
        command += " --wnids";
        string ids;
        for(auto &i: parsMtch.imageNetIDs){
            ids += " " + i;
        }
        command += ids;
    }
    if(!parsMtch.imageNetBuzzWrds.empty()){
        command += " --keywords";
        string wrds;
        for(auto &i: parsMtch.imageNetBuzzWrds){
            wrds += " " + i;
        }
        command += wrds;
    }
    command += " --nr_imgs " + std::to_string(parsMtch.nrImgsFromImageNet);

    cout << "Downloading images from ImageNet. This may take some time ..." << endl;
    try {
        named_mutex mutex(open_or_create, "get_imagenet");
        scoped_lock<named_mutex> lock(mutex);
        int ret = system(command.c_str());
        if (ret) {
            cerr << "Unable to download images from imagenet using command " << command << endl;
            return false;
        }
    }catch(interprocess_exception &ex){
        cerr << ex.what() << std::endl;
        return false;
    }

    vector<string> imgnetdirs = getDirs(imgNetP);
    if(imgnetdirs.empty()){
        return false;
    }
    for(auto &i: imgnetdirs){
        string full_dir = concatPath(imgNetP, i);
        if(i == "xml_files") continue;
        vector<string> filenames1;
        if(!loadImageSequenceNew(full_dir, "/", filenames1)){
            continue;
        }
        if(!filenames1.empty()){
            filenames.insert(filenames.end(), filenames1.begin(), filenames1.end());
        }
    }
    return !filenames.empty();
}

//Calculate number of TP and TN correspondences of the whole sequence
void genMatchSequ::totalNrCorrs() {
    nrCorrsFullSequ = 0;
    for (auto &i : nrCorrs) {
        nrCorrsFullSequ += i;
    }
    if(sequParsLoaded){
        nrTPFullSequ = 0;
        for(size_t i = 0; i < nrCorrs.size(); ++i){
            nrTPFullSequ += static_cast<size_t>(round(inlRat[i] * static_cast<double>(nrCorrs[i])));
        }
        nrTNFullSequ = nrCorrsFullSequ - nrTPFullSequ;
    }else {
        nrTNFullSequ = 0;
        for (auto &i : nrTrueNeg) {
            nrTNFullSequ += i;
        }
        nrTPFullSequ = nrCorrsFullSequ - nrTNFullSequ;
    }
}

bool genMatchSequ::check_3rdPty_GT(){
    use_3dPrtyGT = ((parsMtch.oxfordGTMportion > 0) ? GT_DATASETS::OXFORD:0) |
            ((parsMtch.kittiGTMportion > 0) ? GT_DATASETS::KITTI:0) |
            ((parsMtch.megadepthGTMportion > 0) ? GT_DATASETS::MEGADEPTH:0);
    if (use_3dPrtyGT){
        onlyWarped = false;
        if(use_3dPrtyGT > 2 && use_3dPrtyGT != 4){
            if(nearZero(parsMtch.GTMportion)){
                cerr << "Overall portion of GT feature matches from 3rd party datasets must be set "
                        "if more than 1 dataset is used!. Disabling use of 3rd party datasets." << endl;
                use_3dPrtyGT = 0;
                return false;
            }else{
                double sumpor = parsMtch.oxfordGTMportion + parsMtch.kittiGTMportion + parsMtch.megadepthGTMportion;
                parsMtch.oxfordGTMportion = parsMtch.GTMportion * parsMtch.oxfordGTMportion / sumpor;
                parsMtch.kittiGTMportion = parsMtch.GTMportion * parsMtch.kittiGTMportion / sumpor;
                parsMtch.megadepthGTMportion = parsMtch.GTMportion * parsMtch.megadepthGTMportion / sumpor;
                return true;
            }
        }else{
            switch (use_3dPrtyGT) {
                case GT_DATASETS::OXFORD:
                    parsMtch.GTMportion = parsMtch.oxfordGTMportion;
                    return true;
                case GT_DATASETS::KITTI:
                    parsMtch.GTMportion = parsMtch.kittiGTMportion;
                    return true;
                case GT_DATASETS::MEGADEPTH:
                    parsMtch.GTMportion = parsMtch.megadepthGTMportion;
                    return true;
                default:
                    parsMtch.GTMportion = 0;
                    use_3dPrtyGT = 0;
                    return false;
            }
        }
    }
    onlyWarped = true;
    return false;
}

//Generate GTM from 3rd party datasets
bool genMatchSequ::calcGTM(){
    using namespace boost::interprocess;
    if(!check_3rdPty_GT()){
        resetGTMuse();
        return false;
    }
    try {
        named_mutex mutex(open_or_create, "get_gtm");
        scoped_lock<named_mutex> lock(mutex);
        baseMatcher bm(parsMtch.keyPointType, parsMtch.imgPath, parsMtch.descriptorType, verbose, &rand2);
        bool data_av = false;
        if (use_3dPrtyGT & GT_DATASETS::OXFORD) {
            auto min_nrTP = static_cast<size_t>(round(parsMtch.oxfordGTMportion * static_cast<double>(nrTPFullSequ)));
            cout << "Calculating GTM from Oxford dataset. This may take some time ..." << endl;
            if (!bm.calcGTM_Oxford(min_nrTP)) {
                cerr << "Unable to use GTM from the Oxford dataset." << endl;
            } else {
                data_av |= true;
            }
        }
        if (use_3dPrtyGT & GT_DATASETS::KITTI) {
            auto min_nrTP = static_cast<size_t>(round(parsMtch.kittiGTMportion * static_cast<double>(nrTPFullSequ)));
            cout << "Calculating GTM from KITTI dataset. This may take some time ..." << endl;
            if (!bm.calcGTM_KITTI(min_nrTP)) {
                cerr << "Unable to use GTM from the KITTI dataset." << endl;
            } else {
                data_av |= true;
            }
        }
        if (use_3dPrtyGT & GT_DATASETS::MEGADEPTH) {
            auto min_nrTP = static_cast<size_t>(round(
                    parsMtch.megadepthGTMportion * static_cast<double>(nrTPFullSequ)));
            cout << "Calculating GTM from MegaDepth dataset. This may take some time ..." << endl;
            if (!bm.calcGTM_MegaDepth(min_nrTP, parsMtch.CeresCPUcnt)) {
                cerr << "Unable to use GTM from the MegaDepth dataset." << endl;
            } else {
                data_av |= true;
            }
        }
        if (!data_av) {
            resetGTMuse();
            return false;
        }
        gtmdata = bm.moveGTMdata();
    }catch(interprocess_exception &ex){
        cerr << ex.what() << std::endl;
        return false;
    }
    if(!getGtmDescriptors()){
        resetGTMuse();
        return false;
    }
    gtmdata.matchesGTAllIdx.clear();
    size_t nr_gtm = gtmdata.useNrTP;
//    for(auto &i: gtmdata.matchesGTAll){
//        nr_gtm += i.size();
//    }
    gtmdata.matchesGTAllIdx.reserve(nr_gtm);
    unsigned char sameIdx = 3;
    size_t remTP = 0;
    for(size_t i = 0; i < gtmdata.matchesGTAll.size(); ++i){
        int ii = static_cast<int>(i);
        size_t nrToUse = 0;
        if(ii >= gtmdata.vecIdxRng_Oxford.first && ii < gtmdata.vecIdxRng_Oxford.second){
            if(sameIdx != 0){
                remTP = gtmdata.minNrTP_Oxford;
                sameIdx = 0;
            }
            if(remTP == 0) continue;
            nrToUse = gtmdata.matchesGTAll[i].size();
            if(nrToUse >= remTP){
                nrToUse = remTP;
                remTP = 0;
            }else{
                remTP -= nrToUse;
            }
        }else if(ii >= gtmdata.vecIdxRng_KITTI.first && ii < gtmdata.vecIdxRng_KITTI.second){
            if(sameIdx != 1){
                remTP = gtmdata.minNrTP_KITTI;
                sameIdx = 1;
            }
            if(remTP == 0) continue;
            nrToUse = gtmdata.matchesGTAll[i].size();
            if(nrToUse >= remTP){
                nrToUse = remTP;
                remTP = 0;
            }else{
                remTP -= nrToUse;
            }
        }else if(ii >= gtmdata.vecIdxRng_MegaDepth.first && ii < gtmdata.vecIdxRng_MegaDepth.second){
            if(sameIdx != 2){
                remTP = gtmdata.minNrTP_MegaDepth;
                sameIdx = 2;
            }
            if(remTP == 0) continue;
            nrToUse = gtmdata.matchesGTAll[i].size();
            if(nrToUse >= remTP){
                nrToUse = remTP;
                remTP = 0;
            }else{
                remTP -= nrToUse;
            }
        }else{
            throw SequenceException("Undefined GT dataset");
        }
        for(size_t j = 0; j < nrToUse; ++j){
            gtmdata.matchesGTAllIdx.emplace_back(i, j);
        }
    }
    if(nr_gtm != gtmdata.matchesGTAllIdx.size()){
        throw SequenceException("Number of correspondences in GT datasets do not match");
    }
    vector<std::size_t> idx;
    shuffleVector(idx, nr_gtm);
    reOrderVector(gtmdata.matchesGTAllIdx, idx);
    gtmdata.sum_TN = static_cast<size_t>(floor(0.75 * static_cast<double>(gtmdata.sum_TN)));
    EstimateNrTPTN();
    return true;
}

bool checkMatchesInvalidValues(const std::vector<cv::DMatch> &matches,
                               const std::vector<cv::KeyPoint> &kp1,
                               const std::vector<cv::KeyPoint> &kp2,
                               const cv::Size &img1Size,
                               const cv::Size &img2Size){
    //Check matches
    float i1w, i1h, i2w, i2h;
    i1w = static_cast<float>(img1Size.width) - 2.f;
    i1h = static_cast<float>(img1Size.height) - 2.f;
    i2w = static_cast<float>(img2Size.width) - 2.f;
    i2h = static_cast<float>(img2Size.height) - 2.f;
    const size_t kp1_si = kp1.size();
    const size_t kp2_si = kp2.size();
    const auto kp1_sii = static_cast<int>(kp1_si);
    const auto kp2_sii = static_cast<int>(kp2_si);
    for(auto matche : matches){
        if(matche.queryIdx >= kp1_sii || matche.queryIdx < 0 || matche.trainIdx >= kp2_sii || matche.trainIdx < 0){
            return false;
        }
    }
    for(size_t i = 0; i < kp1_si; i++){
        if(isnan(kp1[i].pt.x) ||
        isnan(kp1[i].pt.y) ||
        isinf(kp1[i].pt.x) ||
        isinf(kp1[i].pt.y) ||
        nearZero(kp1[i].pt.x) ||
        nearZero(kp1[i].pt.y) ||
        kp1[i].pt.x > i1w ||
        kp1[i].pt.y > i1h){
            return false;
        }
    }
    for(size_t i = 0; i < kp2_si; i++){
        if(isnan(kp2[i].pt.x) ||
           isnan(kp2[i].pt.y) ||
           isinf(kp2[i].pt.x) ||
           isinf(kp2[i].pt.y) ||
           nearZero(kp2[i].pt.x) ||
           nearZero(kp2[i].pt.y) ||
           kp2[i].pt.x > i2w ||
           kp2[i].pt.y > i2h){
            return false;
        }
    }
    return true;
}

bool genMatchSequ::getGtmDescriptors(){
    size_t nr_Elems = gtmdata.matchesGTAll.size();
    gtmdata.leftDescriptorsAll.clear();
    gtmdata.leftDescriptorsAll.resize(nr_Elems);
    gtmdata.rightDescriptorsAll.clear();
    gtmdata.rightDescriptorsAll.resize(nr_Elems);
    for (size_t i = 0; i < nr_Elems; ++i) {
        Mat img1 = cv::imread(gtmdata.imgNamesAll[i].first, IMREAD_GRAYSCALE);
        Mat img2 = cv::imread(gtmdata.imgNamesAll[i].second, IMREAD_GRAYSCALE);
        size_t nr_before = gtmdata.keypLAll[i].size();
        vector<KeyPoint> kp_tmp = gtmdata.keypLAll[i];
        if (matchinglib::getDescriptors(img1,
                                        gtmdata.keypLAll[i],
                                        parsMtch.descriptorType,
                                        gtmdata.leftDescriptorsAll[i],
                                        parsMtch.keyPointType) != 0) {
            return false;
        }
//        bool showMatches = false;
        if(nr_before != gtmdata.keypLAll[i].size()){
            if(!recalcMatchIdxs(kp_tmp, i, true)) return false;
//            showMatches = true;
        }else{
            reOrderMatches1Img(kp_tmp, i, true);
        }

        nr_before = gtmdata.keypRAll[i].size();
        kp_tmp = gtmdata.keypRAll[i];
        if (matchinglib::getDescriptors(img2,
                                        gtmdata.keypRAll[i],
                                        parsMtch.descriptorType,
                                        gtmdata.rightDescriptorsAll[i],
                                        parsMtch.keyPointType) != 0) {
            return false;
        }
        if(nr_before != gtmdata.keypRAll[i].size()){
            if(!recalcMatchIdxs(kp_tmp, i, false)) return false;
//            showMatches = true;
        }else{
            reOrderMatches1Img(kp_tmp, i, false);
        }
        if(!checkMatchesInvalidValues(gtmdata.matchesGTAll[i],
                                      gtmdata.keypLAll[i],
                                      gtmdata.keypRAll[i],
                                      img1.size(),
                                      img2.size())){
            return false;
        }

//        kp_tmp = gtmdata.keypLAll[i];
//        Mat descr = gtmdata.leftDescriptorsAll[i].clone();
//        gtmdata.leftDescriptorsAll[i].release();
//        nr_before = gtmdata.keypLAll[i].size();
//        if (matchinglib::getDescriptors(img1,
//                                        gtmdata.keypLAll[i],
//                                        parsMtch.descriptorType,
//                                        gtmdata.leftDescriptorsAll[i],
//                                        parsMtch.keyPointType) != 0) {
//            return false;
//        }
//        if(nr_before != gtmdata.keypLAll[i].size()){
//            throw SequenceException("During this step the number of descriptors should remain the same");
//        }else{
//            reOrderMatches1Img(kp_tmp, i, true);
//        }
//        for (int j = 0; j < gtmdata.leftDescriptorsAll[i].rows; ++j) {
//            double dist = getDescriptorDistance(gtmdata.leftDescriptorsAll[i].row(j), descr.row(j));
//            if(!nearZero(dist)){
//                cerr << "Error" << endl;
//            }
//        }
//        if(showMatches){
//            showMatchesGTMpart(gtmdata.matchesGTAll[i],
//                               gtmdata.leftInlierAll[i],
//                               gtmdata.keypLAll[i],
//                               gtmdata.keypRAll[i],
//                               img1,
//                               img2);
//        }
    }
    return true;
}

void showMatchesGTMpart(const std::vector<cv::DMatch> &matchesGT,
                        const std::vector<bool> &leftInlier,
                        const std::vector<cv::KeyPoint> &keypL,
                        const std::vector<cv::KeyPoint> &keypR,
                        const cv::Mat &img1,
                        const cv::Mat &img2){
    Mat img_match;
    std::vector<cv::KeyPoint> keypL_reduced;//Left keypoints
    std::vector<cv::KeyPoint> keypR_reduced;//Right keypoints
    std::vector<cv::DMatch> matches_reduced;
    std::vector<cv::KeyPoint> keypL_reduced1;//Left keypoints
    std::vector<cv::KeyPoint> keypR_reduced1;//Right keypoints
    std::vector<cv::DMatch> matches_reduced1;
    int j = 0;
    size_t keepNMatches = 100;
    size_t keepXthMatch = 1;
    if(matchesGT.size() > keepNMatches)
        keepXthMatch = matchesGT.size() / keepNMatches;
    for (auto & i : matchesGT)
    {
        int idx = i.queryIdx;
        if(leftInlier[idx])
        {
            keypL_reduced.push_back(keypL[idx]);
            matches_reduced.push_back(i);
            matches_reduced.back().queryIdx = j;
            keypR_reduced.push_back(keypR[matches_reduced.back().trainIdx]);
            matches_reduced.back().trainIdx = j;
            j++;
        }else{
            cout << "Outlier in GTM" << endl;
        }
    }
    j = 0;
    for (unsigned int i = 0; i < matches_reduced.size(); i++)
    {
        if((i % (int)keepXthMatch) == 0)
        {
            keypL_reduced1.push_back(keypL_reduced[i]);
            matches_reduced1.push_back(matches_reduced[i]);
            matches_reduced1.back().queryIdx = j;
            keypR_reduced1.push_back(keypR_reduced[i]);
            matches_reduced1.back().trainIdx = j;
            j++;
        }
    }
    drawMatches(img1, keypL_reduced1, img2, keypR_reduced1, matches_reduced1, img_match);
    imshow("Ground truth matches", img_match);
    waitKey(0);
    cv::destroyAllWindows();
}

void genMatchSequ::reOrderMatches1Img(const std::vector<cv::KeyPoint> &kpsBefore, const size_t &idx, bool isLeftKp){
    std::vector<cv::KeyPoint> *kpsAfter;
    Mat *descriptors;
    if(isLeftKp){
        kpsAfter = &(gtmdata.keypLAll[idx]);
        descriptors = &gtmdata.leftDescriptorsAll[idx];
    }else{
        kpsAfter = &(gtmdata.keypRAll[idx]);
        descriptors = &gtmdata.rightDescriptorsAll[idx];
    }
    KeypointSearch kpS(*kpsAfter);
    vector<int> newToOldIdxs;
    kpS.getNewToOldOrdering(kpsBefore, newToOldIdxs);
    reOrderVector(*kpsAfter, newToOldIdxs);
    reOrderDescriptors(*descriptors, newToOldIdxs);
}

void reOrderDescriptors(cv::Mat &descriptors, const std::vector<int> &idxs){
    CV_Assert((size_t)descriptors.rows == idxs.size());
    cv::Mat descriptor1_tmp;
    descriptor1_tmp.reserve((size_t)descriptors.rows);
    for(int idx : idxs){
        descriptor1_tmp.push_back(descriptors.row(idx));
    }
    descriptor1_tmp.copyTo(descriptors);
}

//Recalculate match indices if the descriptor extraction removed keypoints
bool genMatchSequ::recalcMatchIdxs(const std::vector<cv::KeyPoint> &kpsBefore, const size_t &idx, bool isLeftKp){
    size_t nrdiff = 0;
    std::vector<cv::KeyPoint> *kpsAfter, *kpsAfter2;
    vector<bool> *inl, *inl2;
    Mat *descriptors, *descriptors2;
    if(isLeftKp){
        nrdiff = kpsBefore.size() - gtmdata.keypLAll[idx].size();
        kpsAfter = &(gtmdata.keypLAll[idx]);
        kpsAfter2 = &(gtmdata.keypRAll[idx]);
        inl = &(gtmdata.leftInlierAll[idx]);
        inl2 = &(gtmdata.rightInlierAll[idx]);
        descriptors = &gtmdata.leftDescriptorsAll[idx];
        descriptors2 = &gtmdata.rightDescriptorsAll[idx];
    }else{
        nrdiff = kpsBefore.size() - gtmdata.keypRAll[idx].size();
        kpsAfter = &(gtmdata.keypRAll[idx]);
        kpsAfter2 = &(gtmdata.keypLAll[idx]);
        inl = &(gtmdata.rightInlierAll[idx]);
        inl2 = &(gtmdata.leftInlierAll[idx]);
        descriptors = &gtmdata.rightDescriptorsAll[idx];
        descriptors2 = &gtmdata.leftDescriptorsAll[idx];
    }
    if(nrdiff == 0) return true;
    KeypointSearch kpS(*kpsAfter);
    vector<int> missingIdxs, delIdxs, newToOldIdxs;
    kpS.getNewToOldOrdering(kpsBefore, newToOldIdxs);
    reOrderVector(*kpsAfter, newToOldIdxs);
    reOrderDescriptors(*descriptors, newToOldIdxs);
    kpS.getMissingIdxs(kpsBefore, missingIdxs);
    if(missingIdxs.empty()) return false;
    unordered_set<int> idxs, idxs2;
    for(auto &i: missingIdxs){
        inl->erase(inl->begin() + i);
        idxs.emplace(i);
    }
    vector<int> delKpIdx;
    correctMatchesIdx(gtmdata.matchesGTAll[idx], isLeftKp, idxs, missingIdxs, delKpIdx);
    size_t nrTPdel = delKpIdx.size();
    gtmdata.sum_TP -= nrTPdel;
//    correctMatchesIdx(gtmdata.matchesTNAll[idx], isLeftKp, idxs, missingIdxs, delKpIdx);
//    size_t nrTNdel = delKpIdx.size() - nrTPdel;
    size_t nrTNdel = 0;
    if(nrTPdel != nrdiff){
        nrTNdel = nrdiff - nrTPdel;
    }
    gtmdata.sum_TN -= nrTNdel;
    const int idxI = static_cast<int>(idx);
    if(idxI >= gtmdata.vecIdxRng_Oxford.first && idxI < gtmdata.vecIdxRng_Oxford.second){
        gtmdata.sum_TP_Oxford -= nrTPdel;
        gtmdata.sum_TN_Oxford -= nrTNdel;
        if(gtmdata.minNrTP_Oxford >= gtmdata.sum_TP_Oxford){
            gtmdata.minNrTP_Oxford -= nrTPdel;
            gtmdata.useNrTP -= nrTPdel;
        }
    }else if(idxI >= gtmdata.vecIdxRng_KITTI.first && idxI < gtmdata.vecIdxRng_KITTI.second){
        gtmdata.sum_TP_KITTI -= nrTPdel;
        gtmdata.sum_TN_KITTI -= nrTNdel;
        if(gtmdata.minNrTP_KITTI >= gtmdata.sum_TP_KITTI){
            gtmdata.minNrTP_KITTI -= nrTPdel;
            gtmdata.useNrTP -= nrTPdel;
        }
    }else if(idxI >= gtmdata.vecIdxRng_MegaDepth.first && idxI < gtmdata.vecIdxRng_MegaDepth.second){
        gtmdata.sum_TP_MegaDepth -= nrTPdel;
        gtmdata.sum_TN_MegaDepth -= nrTNdel;
        if(gtmdata.minNrTP_MegaDepth >= gtmdata.sum_TP_MegaDepth){
            gtmdata.minNrTP_MegaDepth -= nrTPdel;
            gtmdata.useNrTP -= nrTPdel;
        }
    }else{
        return false;
    }
    if(!delKpIdx.empty()) {
        for (auto &i: delKpIdx) {
            idxs2.emplace(i);
        }
        sort(delKpIdx.begin(), delKpIdx.end(), [](const int &first, const int &second) { return first > second; });
        for (auto &i: delKpIdx) {
            inl2->erase(inl2->begin() + i);
            kpsAfter2->erase(kpsAfter2->begin() + i);
            if(!descriptors2->empty()) {
                if(i > 0 && i < descriptors2->rows - 1) {
                    Mat b, c, d;
                    descriptors2->rowRange(0, i).copyTo(b);
                    descriptors2->rowRange(i + 1, descriptors2->rows).copyTo(c);
                    vconcat(b, c, d);
                    d.copyTo(*descriptors2);
                }else if(i == 0){
                    descriptors2->rowRange(1, descriptors2->rows).copyTo(*descriptors2);
                }else if(i == descriptors2->rows - 1){
                    descriptors2->rowRange(0, descriptors2->rows - 1).copyTo(*descriptors2);
                }else{
                    throw SequenceException("Wrong number of GTM keypoints or descriptors");
                }
            }
        }
        CV_Assert(descriptors2->empty() || static_cast<size_t>(descriptors2->rows) == kpsAfter2->size());
        vector<int> delKpIdx2;
        correctMatchesIdx(gtmdata.matchesGTAll[idx], !isLeftKp, idxs2, delKpIdx, delKpIdx2);
//        correctMatchesIdx(gtmdata.matchesTNAll[idx], !isLeftKp, idxs2, delKpIdx, delKpIdx2);
        if(!delKpIdx2.empty()) return false;
    }
    return true;
}

void correctMatchesIdx(vector<DMatch> &matches,
        bool isLeftKp,
        const unordered_set<int> &idxs,
        const vector<int> &missingIdxs,
        vector<int> &delKpIdx){
    vector<int> delIdx;
    int idxm = 0;
    for(auto &mtch: matches){
        if(isLeftKp){
            if(correctMatchIdx(idxs, missingIdxs, mtch.queryIdx)){
                delIdx.push_back(idxm);
            }
        }else{
            if(correctMatchIdx(idxs, missingIdxs, mtch.trainIdx)){
                delIdx.push_back(idxm);
            }
        }
        idxm++;
    }
    if(!delIdx.empty()) {
        sort(delIdx.begin(), delIdx.end(), [](const int &first, const int &second) { return first > second; });
        for (auto &i: delIdx) {
            if(isLeftKp){
                delKpIdx.push_back(matches[i].trainIdx);
            }else {
                delKpIdx.push_back(matches[i].queryIdx);
            }
            matches.erase(matches.begin() + i);
        }
    }
}

bool correctMatchIdx(const unordered_set<int> &idxs, const vector<int> &missingIdxs, int &queryTrainIdx){
    if(idxs.find(queryTrainIdx) != idxs.end()){
        return true;
    }
    getNewFromOldIdx(missingIdxs, queryTrainIdx);
    return false;
}

void getNewFromOldIdx(const vector<int> &missingIdxs, int &oldIdx){
    const int mis = static_cast<int>(missingIdxs.size());
    int i1 = 0;
    for (int i = 0; i < mis; ++i) {
        if(oldIdx > missingIdxs[i]){
            i1++;
            if(i == mis - 1){
                oldIdx -= i1;
                if(oldIdx < 0){
                    cerr << "err" << endl;
                }
            }
        }else if(i1 > 0){
            oldIdx -= i1;
            if(oldIdx < 0){
                cerr << "err" << endl;
            }
            break;
        }
    }
}

//Resets all GTM related variables to only use warped patches
void genMatchSequ::resetGTMuse(){
    parsMtch.WarpedPortionTN = 1.0;
    nrTNFullSequWarped = nrTNFullSequ;
    nrGrossTNFullSequWarped = static_cast<size_t>(round(parsMtch.portionGrossTN * static_cast<double>(nrTNFullSequ)));
    nrGrossTNFullSequGTM = 0;
    nrCorrsFullSequWarped = nrCorrsFullSequ;
    nrTPFullSequWarped = nrTPFullSequ;
    //Corresponds to nrCorrsExtractWarped = 2 * nrGrossTNFullSequWarped + (nrTNFullSequWarped - nrGrossTNFullSequWarped) + nrTPFullSequWarped;
    nrCorrsExtractWarped = nrGrossTNFullSequWarped + nrTNFullSequWarped + nrTPFullSequWarped;
    gtmdata.clear();
}

//Update image IDs by adapting. Used if image was not be able to be loaded.
void genMatchSequ::updateImageIDs(const size_t &fromPos, const size_t &toPos, const size_t &len){
    for (size_t i = fromPos; i < fromPos + len; ++i) {
        uniqueImgIDToName.erase(i);
    }
    size_t cnt = fromPos;
    for (size_t i = fromPos + len; i < toPos; ++i) {
        string *tmp = uniqueImgIDToName.at(i);
        uniqueImgIDToName.erase(i);
        uniqueImgIDToName.emplace(cnt++, tmp);
    }
}

//Extracts the necessary number of keypoints from the set of images
bool genMatchSequ::getFeatures() {
    size_t nrImgs = 0;
    size_t kpCnt = 0;
    keypoints1.clear();
    descriptors1.release();
    if(nrCorrsExtractWarped > 0) {
        //Load image names
        if (!getImageList()) {
            return false;
        }
        //Get random sequence of images
        std::shuffle(imageList.begin(), imageList.end(), std::mt19937{std::random_device{}()});
    }
    calcGTMimgIDs();
    if(gtmdata.isValid()){
        nrImgs = gtmdata.getNumberImgs();
    }
    minNrFramesMatch = max(min(minNrFramesMatch, totalNrFrames / 2), static_cast<size_t>(1));
    if(nrCorrsExtractWarped > 0) {
        nrImgs += imageList.size();

        //Check for the correct keypoint & descriptor types
        if (!matchinglib::IsKeypointTypeSupported(parsMtch.keyPointType)) {
            cerr << "Keypoint type " << parsMtch.keyPointType << " is not supported!" << endl;
            return false;
        }
        if (!matchinglib::IsDescriptorTypeSupported(parsMtch.descriptorType)) {
            cerr << "Descriptor type " << parsMtch.descriptorType << " is not supported!" << endl;
            return false;
        }

        //Load images and extract features & descriptors
        int errCnt = 0;
        const int maxErrCnt = 10;
        if (nrImgs <= maxImgLoad) {
            imgs.reserve(nrImgs);
        }
        for (size_t i = 0; i < imageList.size(); ++i) {
            //Load image
            Mat img;
            try {
                if (nrImgs <= maxImgLoad) {
                    imgs.emplace_back(cv::imread(imageList[i], IMREAD_GRAYSCALE));
                    img = imgs.back();
                } else {
                    img = cv::imread(imageList[i], IMREAD_GRAYSCALE);
                }
            } catch (...) {
                if(verbose & PRINT_WARNING_MESSAGES) {
                    cout << "Unable to load image " << imageList[i] << " -> skipping" << endl;
                }
                continue;
            }
//        imgs.emplace_back(cv::imread(imageList[i], IMREAD_GRAYSCALE));
            std::vector<cv::KeyPoint> keypoints1Img;

            //Extract keypoints
            if (matchinglib::getKeypoints(img, keypoints1Img, parsMtch.keyPointType, true, 8000) != 0) {
                errCnt++;
                if (errCnt > maxErrCnt) {
                    cerr << "Extraction of keypoints failed for too many images!" << endl;
                    return false;
                }
            }

            //Compute descriptors
            cv::Mat descriptors1Img;
            if (matchinglib::getDescriptors(img,
                                            keypoints1Img,
                                            parsMtch.descriptorType,
                                            descriptors1Img,
                                            parsMtch.keyPointType) != 0) {
                errCnt++;
                if (errCnt > maxErrCnt) {
                    cout << "Calculation of descriptors failed for too many images!" << endl;
                    return false;
                }
            }
            errCnt = 0;
            CV_Assert(keypoints1Img.size() == (size_t) descriptors1Img.rows);
            keypoints1.insert(keypoints1.end(), keypoints1Img.begin(), keypoints1Img.end());
            if (descriptors1.empty()) {
                descriptors1 = descriptors1Img;
            } else {
                descriptors1.push_back(descriptors1Img);
            }
            kpCnt += keypoints1Img.size();
            vector<size_t> imgNr_tmp = vector<size_t>(keypoints1Img.size(), i);
            featureImgIdx.insert(featureImgIdx.end(), imgNr_tmp.begin(), imgNr_tmp.end());
            if (kpCnt > (size_t) (1.03 * (double) nrCorrsExtractWarped)) {
                break;
            }
        }
        //Update number of available features
        size_t kpCntDiff = kpCnt - nrCorrsExtractWarped;
        nrCorrsFullSequWarped += kpCntDiff;
        double inlRatTmp = static_cast<double>(nrTPFullSequ) / static_cast<double>(nrCorrsFullSequ);
        auto tp_part = static_cast<size_t>(round(inlRatTmp * static_cast<double>(kpCntDiff)));
        auto tn_part = kpCntDiff - tp_part;
        auto gross_part = static_cast<size_t>(floor(parsMtch.portionGrossTN * static_cast<double>(tn_part) / 2.));
        nrTNFullSequWarped += tn_part - gross_part;
        nrGrossTNFullSequWarped += gross_part;
        nrTPFullSequWarped += tp_part;
        nrCorrsExtractWarped = kpCnt;
    }else{
        if (nrImgs <= maxImgLoad) {
            imgs.reserve(nrImgs);
        }
    }

    //Calculate statistics for bad descriptor distances
    if(!calcGoodBadDescriptorTH()){
        return false;
    }

    if(nrCorrsExtractWarped > 0) {
        //Shuffle keypoints and descriptors
        vector<KeyPoint> keypoints1_tmp(keypoints1.size());
        Mat descriptors1_tmp;
        descriptors1_tmp.reserve((size_t) descriptors1.rows);
        vector<size_t> featureImgIdx_tmp(featureImgIdx.size());
        vector<size_t> featureIdxs;
        shuffleVector(featureIdxs, keypoints1.size());
        reOrderVector(keypoints1, featureIdxs);
        reOrderVector(featureImgIdx, featureIdxs);
        for (auto &i : featureIdxs) {
            descriptors1_tmp.push_back(descriptors1.row((int) i));
        }
        descriptors1_tmp.copyTo(descriptors1);
    }

    //Generate index for correspondences
    buildCorrsIdx();

    kpCnt = corrToIdxMap.size();

    //Init index to keypoints, descriptors and featureImgIdx with optionally containing duplicates
    bool repStereo = !(nearZero(parsMtch.repeatPatternPortStereo.first) && nearZero(parsMtch.repeatPatternPortStereo.second));
    bool repFrame = !(nearZero(parsMtch.repeatPatternPortFToF.first) && nearZero(parsMtch.repeatPatternPortFToF.second));
    if(!repFrame && !repStereo) {
        featureIdxRepPatt = vector<size_t>(kpCnt);
        std::iota(featureIdxRepPatt.begin(), featureIdxRepPatt.end(), 0);
    }else{
        size_t kpCnt2 = 0, sum_idxs = 0;
        vector<size_t> kpPtr(kpCnt);
        std::iota(kpPtr.begin(), kpPtr.end(), 0);
        double repPortF = 0;
        bool repPortF_var = false;
        if(repFrame){
            if(nearZero(parsMtch.repeatPatternPortFToF.first - parsMtch.repeatPatternPortFToF.second)){
                repPortF = parsMtch.repeatPatternPortFToF.first;
            }else{
                repPortF_var = true;
                repPortF = getRandDoubleValRng(parsMtch.repeatPatternPortFToF.first, parsMtch.repeatPatternPortFToF.second);
            }
        }
        for(size_t i = 0; i < totalNrFrames; ++i){
            double repPortS = 0;
            vector<size_t> kpPtr_tmp;
            size_t repsF_size = 0;
            if(!repStereo && repFrame && (i == 0)){
                if(nrCorrs[0] > kpCnt){
                    return false;
                }
                featureIdxRepPatt = vector<size_t>(nrCorrs[0]);
                std::iota(featureIdxRepPatt.begin(), featureIdxRepPatt.end(), 0);
                kpCnt2 = nrCorrs[0];
                continue;
            }
            //Get indices for repeated patterns from frame to frame
            if(repFrame){
                if(repPortF_var){
                    repPortF = getRandDoubleValRng(parsMtch.repeatPatternPortFToF.first, parsMtch.repeatPatternPortFToF.second);
                }
                if(i == 0){
                    kpPtr_tmp = vector<size_t>(nrCorrs[0]);
                    std::iota(kpPtr_tmp.begin(), kpPtr_tmp.end(), 0);
                    kpCnt2 = nrCorrs[0];
                    repsF_size = 0;
                    sum_idxs = nrCorrs[0];
                }else{
                    vector<size_t> uniquesF, repsF;
                    auto uniqueKpF = static_cast<size_t>(std::round(static_cast<double>(nrCorrs[i]) * (1.0 - repPortF)));
                    size_t repKpF = nrCorrs[i] - uniqueKpF;
                    if((kpCnt2 + uniqueKpF) > kpCnt){
                        kpCnt = kpCnt2;
                        break;
                    }
                    size_t uniEndF = kpCnt2 + uniqueKpF;
                    if(uniqueKpF > 0) {
                        uniquesF.insert(uniquesF.end(), kpPtr.begin() + kpCnt2, kpPtr.begin() + uniEndF);
                    }
                    kpCnt2 = uniEndF;
                    if(repKpF == 0){
                        kpPtr_tmp = uniquesF;
                        repsF_size = 0;
                    }else if(repKpF < 6){
                        kpPtr_tmp.resize(kpPtr_tmp.size() + repKpF,
                                featureIdxRepPatt[featureIdxRepPatt.size() - nrCorrs[i - 1] + (rand2() % (nrCorrs[i - 1]))]);
                        if(uniqueKpF > 0) {
                            kpPtr_tmp.insert(kpPtr_tmp.end(), uniquesF.begin(), uniquesF.end());
                        }
                        repsF_size = repKpF;
                    }else{
                        size_t maxSimilarF = std::max(static_cast<size_t>(rand2() % (repKpF / 3)), static_cast<size_t>(1));
                        std::uniform_int_distribution<size_t> distributionF;
                        if(maxSimilarF > 1) {
                            distributionF = std::uniform_int_distribution<size_t>(1, maxSimilarF);
                        }
                        const size_t idxF_max = nrCorrs[i - 1];
                        size_t rep_cntF = 0, idxF = 0;
                        vector<size_t> idx_tmp(nrCorrs[i - 1]);
                        std::iota(idx_tmp.begin(), idx_tmp.end(), featureIdxRepPatt.size() - nrCorrs[i - 1]);
                        std::shuffle(idx_tmp.begin(), idx_tmp.end(), rand2);
                        while(rep_cntF < repKpF){
                            idxF++;
                            if(idxF == idxF_max){
                                repsF.resize(repsF.size() + repKpF - rep_cntF, featureIdxRepPatt[idx_tmp[idxF - 1]]);
                                break;
                            }
                            size_t nrReps = 1;
                            if(maxSimilarF > 1) {
                                nrReps = distributionF(rand_gen);
                            }
                            if((rep_cntF + nrReps) > repKpF){
                                nrReps = repKpF - rep_cntF;
                            }
                            repsF.resize(repsF.size() + nrReps, featureIdxRepPatt[idx_tmp[idxF - 1]]);
                            rep_cntF += nrReps;
                        }
                        kpPtr_tmp.insert(kpPtr_tmp.end(), repsF.begin(), repsF.end());
                        if(uniqueKpF > 0) {
                            kpPtr_tmp.insert(kpPtr_tmp.end(), uniquesF.begin(), uniquesF.end());
                        }
                        repsF_size = repsF.size();
                    }
                    sum_idxs += nrCorrs[i];
                }
            }else{
                kpPtr_tmp = vector<size_t>(nrCorrs[i]);
                std::iota(kpPtr_tmp.begin(), kpPtr_tmp.end(), kpCnt2);
            }

            //Get indices for repeated patterns inbetween stereo frames
            if(repStereo) {
                if (nearZero(parsMtch.repeatPatternPortStereo.first - parsMtch.repeatPatternPortStereo.second)) {
                    repPortS = parsMtch.repeatPatternPortStereo.first;
                } else {
                    repPortS = getRandDoubleValRng(parsMtch.repeatPatternPortStereo.first,
                                                   parsMtch.repeatPatternPortStereo.second);
                }

                auto uniqueKpS = static_cast<size_t>(std::round(static_cast<double>(nrCorrs[i]) * (1.0 - repPortS)));
                size_t repKpS = nrCorrs[i] - uniqueKpS;
                if (!repFrame && ((kpCnt2 + uniqueKpS + repKpS / 2) >= kpCnt)) {
                    kpCnt = kpCnt2;
                    break;
                }
                //size_t kpCnt2S = 0;
                //size_t uniEndS = kpCnt2 + uniqueKpS;
                vector<size_t> uniquesS;
                vector<size_t> repsS;
                if(uniqueKpS > 0) {
                    uniquesS.insert(uniquesS.end(), kpPtr_tmp.begin(), kpPtr_tmp.begin() + uniqueKpS);
                }
                if (repKpS == 0) {
                    featureIdxRepPatt.insert(featureIdxRepPatt.end(), uniquesS.begin(), uniquesS.end());
                    if(!repFrame) {
                        kpCnt2 += nrCorrs[i];
                        sum_idxs += nrCorrs[i];
                    }
                    continue;
                } else if (repKpS == 1) {
                    uniquesS.push_back(kpPtr_tmp[uniqueKpS]);
                    featureIdxRepPatt.insert(featureIdxRepPatt.end(), uniquesS.begin(), uniquesS.end());
                    if(!repFrame) {
                        kpCnt2 += nrCorrs[i];
                        sum_idxs += nrCorrs[i];
                    }
                    continue;
                } else if (repKpS < 9) {
                    if(uniqueKpS > 0) {
                        featureIdxRepPatt.insert(featureIdxRepPatt.end(), uniquesS.begin(), uniquesS.end());
                    }
                    featureIdxRepPatt.resize(featureIdxRepPatt.size() + repKpS, kpPtr_tmp[uniqueKpS]);
                    if(!repFrame) {
                        std::shuffle(featureIdxRepPatt.begin() + sum_idxs, featureIdxRepPatt.end(), rand2);
                        kpCnt2 += uniqueKpS + 1;
                        sum_idxs += nrCorrs[i];
                    }else{
                        std::shuffle(featureIdxRepPatt.begin() + sum_idxs - nrCorrs[i], featureIdxRepPatt.end(), rand2);
                    }
                    continue;
                }
                size_t maxSimilarS = std::max(static_cast<size_t>(rand2() % (repKpS / 3)), static_cast<size_t>(3));
                if(repsF_size > 0){
                    if( nrCorrs[i] / maxSimilarS < repsF_size){
                        maxSimilarS = max(nrCorrs[i] / repsF_size, static_cast<size_t>(2));
                    }
                }
                std::uniform_int_distribution<size_t> distributionS;
                if(maxSimilarS > 2){
                    distributionS = std::uniform_int_distribution<size_t>(2, maxSimilarS);
                }
                size_t rep_cntS = 0, idxS = uniqueKpS;
                while (rep_cntS < repKpS) {
                    idxS++;
                    if (idxS == nrCorrs[i]) {
                        repsS.resize(repsS.size() + repKpS - rep_cntS, kpPtr_tmp[idxS - 1]);
                        break;
                    }
                    size_t nrReps = 2;
                    if(maxSimilarS > 2){
                        nrReps = distributionS(rand_gen);
                    }
                    if ((rep_cntS + nrReps) > repKpS) {
                        nrReps = repKpS - rep_cntS;
                    }
                    repsS.resize(repsS.size() + nrReps, kpPtr_tmp[idxS - 1]);
                    rep_cntS += nrReps;
                }
                if(uniqueKpS > 0) {
                    featureIdxRepPatt.insert(featureIdxRepPatt.end(), uniquesS.begin(), uniquesS.end());
                }
                featureIdxRepPatt.insert(featureIdxRepPatt.end(), repsS.begin(), repsS.end());
                if(!repFrame) {
                    std::shuffle(featureIdxRepPatt.begin() + sum_idxs, featureIdxRepPatt.end(), rand2);
                    kpCnt2 += idxS;
                    sum_idxs += nrCorrs[i];
                }else{
                    std::shuffle(featureIdxRepPatt.begin() + sum_idxs - nrCorrs[i], featureIdxRepPatt.end(), rand2);
                    if(idxS <= repsF_size){
                        kpCnt2 -= nrCorrs[i] - repsF_size;
                    }else{
                        kpCnt2 -= nrCorrs[i] - repsF_size - idxS;
                    }
                }
            }else{
                featureIdxRepPatt.insert(featureIdxRepPatt.end(), kpPtr_tmp.begin(), kpPtr_tmp.end());
            }
        }
        if(featureImgIdx.size() > sum_idxs){
            featureIdxRepPatt.insert(featureIdxRepPatt.end(), kpPtr.begin() + kpCnt2, kpPtr.end());
        }
        kpCnt = featureIdxRepPatt.size();
    }

    if (kpCnt < nrCorrsFullSequ) {
        cout << "Too less keypoints - please provide additional images! "
        << nrCorrsFullSequ << " features are required but only "
        << kpCnt << " could be extracted from the images." << endl;
        if (parsMtch.takeLessFramesIfLessKeyP) {
            nrCorrsFullSequ = 0;
            size_t i = 0;
            for (; i < nrCorrs.size(); ++i) {
                if ((kpCnt - nrCorrs[i]) > nrCorrsFullSequ) {
                    nrCorrsFullSequ += nrCorrs[i];
                } else {
                    break;
                }
            }
            nrFramesGenMatches = i;
            if (nrFramesGenMatches < minNrFramesMatch) {
                cout << "Only able to generate matches for " << nrFramesGenMatches <<
                     " frames but a minimum of " << minNrFramesMatch <<
                     " would be required." << endl;
                return false;
            } else {
                cout << "Calculating matches for only " << nrFramesGenMatches <<
                     " out of " << totalNrFrames << " frames." << endl;
            }
        } else {
            return false;
        }
    } else {
        nrFramesGenMatches = totalNrFrames;
    }

    //Prepare map of most used images ( calculation is performed in getMostUsedImgs() )
    imgFrameIdxMap.resize(nrFramesGenMatches);
    if(nrImgs > maxImgLoad) {
        loadImgsEveryFrame = true;
    }else{
        loadImgsEveryFrame = false;
        size_t idx = 0;
        for(auto &i: uniqueImgIDToName){
            imgFrameIdxMap[0].first[i.first] = idx;
            imgFrameIdxMap[0].second.push_back(i.first);
            idx++;
        }
        for(auto& i : imgFrameIdxMap[0].second){
            imgs.emplace_back(cv::imread(*uniqueImgIDToName.at(i), IMREAD_GRAYSCALE));
        }
        imgFrameIdxMap.insert(imgFrameIdxMap.begin() + 1, nrFramesGenMatches - 1, imgFrameIdxMap[0]);
    }

    return true;
}

void genMatchSequ::buildCorrsIdx(){
    corrToIdxMap.clear();
    vector<uint8_t> corrTypeIdx;
    if(nrTPFullSequWarped > 0){
        corrTypeIdx = vector<uint8_t>(nrTPFullSequWarped, 0);
    }
    if(nrGrossTNFullSequWarped > 0){
        corrTypeIdx.insert(corrTypeIdx.end(), nrGrossTNFullSequWarped, 1);
    }
    size_t nrTNwarpSingle = nrTNFullSequWarped - nrGrossTNFullSequWarped;
    CV_Assert(nrTNwarpSingle <= nrTNFullSequWarped);
    if(nrTNwarpSingle > 0){
        corrTypeIdx.insert(corrTypeIdx.end(), nrTNwarpSingle, 2);
    }
    if(gtmdata.isValid()){
        //Generate IDs for images
        corrTypeIdx.insert(corrTypeIdx.end(), gtmdata.useNrTP, 3);
        if(nrGrossTNFullSequGTM > 0){
            corrTypeIdx.insert(corrTypeIdx.end(), nrGrossTNFullSequGTM, 4);
        }
    }
    vector<std::size_t> idx;
    shuffleVector(idx, corrTypeIdx.size());
    reOrderVector(corrTypeIdx, idx);
    size_t id_run = 0, idx_warped = 0, idx_gtm_TP = 0, idx_gtm_TN = 0;
    vector<uint8_t> type_cache;
    double descr_norm = 0;
    size_t gtm_idx1 = 0, gtm_idx2 = 0;
    DMatch *match_tmp;
    for(auto &i: corrTypeIdx){
        type_cache.push_back(i);
        bool skip = false;
        while(!type_cache.empty()) {
            switch (type_cache.back()) {
                case 0:
                    corrToIdxMap.emplace(id_run, KeypointIndexer(false,
                                                                 &imageList[featureImgIdx[idx_warped]],
                                                                 descriptors1,
                                                                 static_cast<int>(idx_warped),
                                                                 &keypoints1[idx_warped],
                                                                 id_run,
                                                                 featureImgIdx[idx_warped]));
                    idx_warped++;
                    tntpindexer.addTPID(id_run);
                    type_cache.pop_back();
                    break;
                case 1:
                    //Check descriptor distance
                    if(descriptors1.type() == CV_8U) {
                        descr_norm = norm(descriptors1.row(static_cast<int>(idx_warped)),
                                          descriptors1.row(static_cast<int>(idx_warped + 1)), NORM_HAMMING);
                    }else{
                        descr_norm = norm(descriptors1.row(static_cast<int>(idx_warped)),
                                          descriptors1.row(static_cast<int>(idx_warped + 1)), NORM_L2);
                    }
                    if(descr_norm < badDescrTH.minVal){
                        skip = true;
                        break;
                    }
                    corrToIdxMap.emplace(id_run, KeypointIndexer(&imageList[featureImgIdx[idx_warped]],
                                                                 &imageList[featureImgIdx[idx_warped + 1]],
                                                                 descriptors1,
                                                                 descriptors1,
                                                                 static_cast<int>(idx_warped),
                                                                 static_cast<int>(idx_warped + 1),
                                                                 &keypoints1[idx_warped],
                                                                 &keypoints1[idx_warped + 1],
                                                                 id_run,
                                                                 featureImgIdx[idx_warped],
                                                                 featureImgIdx[idx_warped + 1]));
                    idx_warped += 2;
                    tntpindexer.addTNID(id_run);
                    type_cache.pop_back();
                    break;
                case 2:
                    corrToIdxMap.emplace(id_run, KeypointIndexer(true,
                                                                 &imageList[featureImgIdx[idx_warped]],
                                                                 descriptors1,
                                                                 static_cast<int>(idx_warped),
                                                                 &keypoints1[idx_warped],
                                                                 id_run,
                                                                 featureImgIdx[idx_warped]));
                    idx_warped++;
                    tntpindexer.addTNID(id_run);
                    type_cache.pop_back();
                    break;
                case 3:
                    gtm_idx1 = gtmdata.matchesGTAllIdx[idx_gtm_TP].first;
                    gtm_idx2 = gtmdata.matchesGTAllIdx[idx_gtm_TP].second;
                    match_tmp = &(gtmdata.matchesGTAll[gtm_idx1][gtm_idx2]);
                    corrToIdxMap.emplace(id_run, KeypointIndexer(false,
                                                                 &(gtmdata.imgNamesAll[gtm_idx1].first),
                                                                 &(gtmdata.imgNamesAll[gtm_idx1].second),
                                                                 gtmdata.leftDescriptorsAll[gtm_idx1],
                                                                 gtmdata.rightDescriptorsAll[gtm_idx1],
                                                                 match_tmp->queryIdx,
                                                                 match_tmp->trainIdx,
                                                                 &(gtmdata.keypLAll[gtm_idx1][match_tmp->queryIdx]),
                                                                 &(gtmdata.keypRAll[gtm_idx1][match_tmp->trainIdx]),
                                                                 match_tmp,
                                                                 idx_gtm_TP,
                                                                 id_run,
                                                                 gtmdata.getImgIDbyGTMIdx(gtm_idx1, false),
                                                                 gtmdata.getImgIDbyGTMIdx(gtm_idx1, true),
                                                                 gtmdata.ImgNamesToImgID.at(gtmdata.imgNamesAll[gtm_idx1].first)[0],
                                                                 gtmdata.ImgNamesToImgID.at(gtmdata.imgNamesAll[gtm_idx1].second)[0]));
                    idx_gtm_TP++;
                    tntpindexer.addTPID(id_run);
                    type_cache.pop_back();
                    break;
                case 4:
                    gtm_idx1 = gtmdata.matchesTNAllIdx[idx_gtm_TN].first;
                    gtm_idx2 = gtmdata.matchesTNAllIdx[idx_gtm_TN].second;
                    match_tmp = &(gtmdata.matchesTNAll[gtm_idx1][gtm_idx2]);
                    corrToIdxMap.emplace(id_run, KeypointIndexer(true,
                                                                 &(gtmdata.imgNamesAll[gtm_idx1].first),
                                                                 &(gtmdata.imgNamesAll[gtm_idx1].second),
                                                                 gtmdata.leftDescriptorsAll[gtm_idx1],
                                                                 gtmdata.rightDescriptorsAll[gtm_idx1],
                                                                 match_tmp->queryIdx,
                                                                 match_tmp->trainIdx,
                                                                 &(gtmdata.keypLAll[gtm_idx1][match_tmp->queryIdx]),
                                                                 &(gtmdata.keypRAll[gtm_idx1][match_tmp->trainIdx]),
                                                                 match_tmp,
                                                                 idx_gtm_TN,
                                                                 id_run,
                                                                 gtmdata.getImgIDbyGTMIdx(gtm_idx1, false),
                                                                 gtmdata.getImgIDbyGTMIdx(gtm_idx1, true),
                                                                 gtmdata.ImgNamesToImgID.at(gtmdata.imgNamesAll[gtm_idx1].first)[0],
                                                                 gtmdata.ImgNamesToImgID.at(gtmdata.imgNamesAll[gtm_idx1].second)[0]));
                    idx_gtm_TN++;
                    tntpindexer.addTNID(id_run);
                    type_cache.pop_back();
                    break;
                default:
                    throw SequenceException("Undefined correspondence type.");
            }
            if(skip){
                break;
            }
            id_run++;
        }
    }
}

void genMatchSequ::calcGTMimgIDs(){
    for (size_t j = 0; j < imageList.size(); ++j) {
        uniqueImgIDToName.emplace(j, &imageList[j]);
    }
    size_t min_idx = 0;
    if(gtmdata.isValid()){
        //Generate IDs for images
        min_idx = imageList.size();
        size_t idx = min_idx;
        gtmdata.imgIDToImgNamesAll.clear();
        gtmdata.gtmIdxToImgID.clear();
        gtmdata.ImgNamesToImgID.clear();
        for(size_t i = 0; i < gtmdata.imgNamesAll.size(); ++i){
            gtmdata.imgIDToImgNamesAll.emplace(idx, make_pair(i, false));
            if(gtmdata.ImgNamesToImgID.find(gtmdata.imgNamesAll[i].first) == gtmdata.ImgNamesToImgID.end()){
                gtmdata.ImgNamesToImgID.emplace(gtmdata.imgNamesAll[i].first, std::vector<size_t>(1, idx));
                gtmdata.uniqueImgIDTo1ImgName.emplace(idx, &(gtmdata.imgNamesAll[i].first));
            }else{
                gtmdata.ImgNamesToImgID.at(gtmdata.imgNamesAll[i].first).push_back(idx);
            }
            gtmdata.gtmIdxToImgID.emplace(make_pair(i, false), idx++);
            gtmdata.imgIDToImgNamesAll.emplace(idx, make_pair(i, true));
            if(gtmdata.ImgNamesToImgID.find(gtmdata.imgNamesAll[i].second) == gtmdata.ImgNamesToImgID.end()){
                gtmdata.ImgNamesToImgID.emplace(gtmdata.imgNamesAll[i].second, std::vector<size_t>(1, idx));
                gtmdata.uniqueImgIDTo1ImgName.emplace(idx, &(gtmdata.imgNamesAll[i].second));
            }else{
                gtmdata.ImgNamesToImgID.at(gtmdata.imgNamesAll[i].second).push_back(idx);
            }
            gtmdata.gtmIdxToImgID.emplace(make_pair(i, true), idx++);
        }
        uniqueImgIDToName.insert(gtmdata.uniqueImgIDTo1ImgName.begin(), gtmdata.uniqueImgIDTo1ImgName.end());
    }
}

void genMatchSequ::getMostUsedImgs(){
    //Get most used keypoint images per frame to avoid holding all images in memory in the case a huge amount of images is used
    //Thus, only the maxImgLoad images from which the most keypoints for a single frame are used are loaded into memory
    if(loadImgsEveryFrame){
        size_t featureIdx = featureIdxBegin;
        vector<pair<size_t,size_t>> imgUsageFrequ;
        map<size_t,size_t> imgIdx;
        const size_t tn_start = featureIdxBegin + combNrCorrsTP;
        size_t corr_id = tntpindexer.getCorrID(featureIdxRepPatt[featureIdx], false, true);
        KeypointIndexer *kpinfo = &corrToIdxMap.at(corr_id);
        size_t uimgID = kpinfo->getUniqueImgID1();
        imgUsageFrequ.emplace_back(make_pair(uimgID, 1));
        imgIdx[uimgID] = 0;
        size_t idx = 1;
        if(kpinfo->isUniqueImgID2Valid()){
            uimgID = kpinfo->getUniqueImgID2();
            imgUsageFrequ.emplace_back(make_pair(uimgID, 1));
            imgIdx[uimgID] = 1;
            idx++;
        }
        featureIdx++;
        for (size_t j = 1; j < nrCorrs[actFrameCnt]; ++j) {
            corr_id = tntpindexer.getCorrID(featureIdxRepPatt[featureIdx], featureIdx >= tn_start, true);
            kpinfo = &corrToIdxMap.at(corr_id);
            uimgID = kpinfo->getUniqueImgID1();
            if(imgIdx.find(uimgID) != imgIdx.end()){
                imgUsageFrequ[imgIdx[uimgID]].second++;
            }else{
                imgUsageFrequ.emplace_back(make_pair(uimgID, 1));
                imgIdx[uimgID] = idx;
                idx++;
            }
            if(kpinfo->isUniqueImgID2Valid()){
                uimgID = kpinfo->getUniqueImgID2();
                if(imgIdx.find(uimgID) != imgIdx.end()){
                    imgUsageFrequ[imgIdx[uimgID]].second++;
                }else{
                    imgUsageFrequ.emplace_back(make_pair(uimgID, 1));
                    imgIdx[uimgID] = idx;
                    idx++;
                }
            }
            featureIdx++;
        }
        //Sort based on frequency of usage
        sort(imgUsageFrequ.begin(),
             imgUsageFrequ.end(),
             [](pair<size_t,size_t> &first, pair<size_t,size_t> &second){return first.second > second.second;});
        for(size_t j = 0; j < min(maxImgLoad, imgUsageFrequ.size()); ++j){
            imgFrameIdxMap[actFrameCnt].first[imgUsageFrequ[j].first] = j;
            imgFrameIdxMap[actFrameCnt].second.push_back(imgUsageFrequ[j].first);
        }
    }
}

void genMatchSequ::updateLinearIdx(){
    const size_t idxEndTP = featureIdxBegin + combNrCorrsTP;
    const size_t idxEndTN = idxEndTP + combNrCorrsTN;
    for (size_t i = featureIdxBegin; i < idxEndTP; ++i) {
        tntpindexer.getNextTPID(featureIdxRepPatt[i], featureIdxBegin);
    }
    for (size_t i = idxEndTP; i < idxEndTN; ++i) {
        tntpindexer.getNextTNID(featureIdxRepPatt[i], featureIdxBegin);
    }
}

bool genMatchSequ::generateCorrespondingFeatures(){
//    static size_t featureIdxBegin = 0;
    //Load images if not already done
    if(loadImgsEveryFrame){
        imgs.clear();
        imgs.reserve(imgFrameIdxMap[actFrameCnt].second.size());
        for(auto& i : imgFrameIdxMap[actFrameCnt].second){
            imgs.emplace_back(cv::imread(*uniqueImgIDToName.at(i), IMREAD_GRAYSCALE));
        }
    }
    size_t featureIdxBegin_tmp = featureIdxBegin;
    frameKeypoints1.clear();
    frameKeypoints2.clear();
    frameKeypoints1.resize((size_t)combNrCorrsTP);
    frameKeypoints2.resize((size_t)combNrCorrsTP);
    frameDescriptors1.release();
    frameDescriptors2.release();
    frameMatches.clear();
    frameHomographies.clear();
    frameHomographiesCam1.clear();
    srcImgPatchIdxAndKp.clear();
    frameKeypoints2NoErr.clear();
    corrType.clear();
    generateCorrespondingFeaturesTPTN(featureIdxBegin,
                                      false,
                                      frameKeypoints1,
                                      frameKeypoints2,
                                      frameDescriptors1,
                                      frameDescriptors2,
                                      frameMatches,
                                      frameHomographies,
                                      srcImgPatchIdxAndKp,
                                      &frameHomographiesCam1);
    if(frameDescriptors1.rows != frameDescriptors2.rows){
        cerr << "Number of descriptors between images not matching" << endl;
    }
    if(frameDescriptors1.rows != combNrCorrsTP){
        cerr << "Number of descriptors not matching with number of TP" << endl;
    }
    if(frameMatches.size() != (size_t)combNrCorrsTP){
        cerr << "Number of matches not matching with number of TP" << endl;
    }
    if(frameHomographies.size() != (size_t)combNrCorrsTP){
        cerr << "Number of homographies to img2 not matching with number of TP" << endl;
    }
    if(frameHomographiesCam1.size() != (size_t)combNrCorrsTP){
        cerr << "Number of homographies of img1 not matching with number of TP" << endl;
    }
    if(srcImgPatchIdxAndKp.size() != (size_t)combNrCorrsTP){
        cerr << "Number of keypoints and image indices not matching with number of TP" << endl;
    }
    CV_Assert((frameDescriptors1.rows == frameDescriptors2.rows)
    && (frameDescriptors1.rows == combNrCorrsTP)
    && (frameMatches.size() == (size_t)combNrCorrsTP)
    && (frameHomographies.size() == (size_t)combNrCorrsTP)
    && (frameHomographiesCam1.size() == (size_t)combNrCorrsTP)
    && (srcImgPatchIdxAndKp.size() == (size_t)combNrCorrsTP));
    frameInliers = vector<bool>((size_t)combNrCorrsTP, true);
    getErrorFreeKeypoints(frameKeypoints2, frameKeypoints2NoErr);
    featureIdxBegin_tmp += (size_t)combNrCorrsTP;
    CV_Assert(combNrCorrsTP == (finalNrTPStatCorrs + finalNrTPMovCorrs + finalNrTPStatCorrsFromLast + finalNrTPMovCorrsFromLast));
    for (unsigned char j = 0; j < 4; ++j) {
        if(combCorrsImg12TPorder.statTPnew == j){
            corrType.insert(corrType.end(), finalNrTPStatCorrs, 0);
        }else if(combCorrsImg12TPorder.statTPfromLast == j){
            corrType.insert(corrType.end(), finalNrTPStatCorrsFromLast, 2);
        }else if(combCorrsImg12TPorder.movTPnew == j){
            corrType.insert(corrType.end(), finalNrTPMovCorrs, 1);
        }else{
            corrType.insert(corrType.end(), finalNrTPMovCorrsFromLast, 3);
        }
    }
    if(combNrCorrsTN > 0) {
        vector<KeyPoint> frameKPsTN1((size_t)combNrCorrsTN), frameKPsTN2((size_t)combNrCorrsTN);
        Mat frameDescr1TN, frameDescr2TN;
        vector<DMatch> frameMatchesTN;
        vector<Mat> frameHomosTN;
        std::vector<std::pair<std::pair<size_t,cv::KeyPoint>, std::pair<size_t,cv::KeyPoint>>> srcImgPatchIdxAndKpTN;
        generateCorrespondingFeaturesTPTN(featureIdxBegin_tmp,
                                          true,
                                          frameKPsTN1,
                                          frameKPsTN2,
                                          frameDescr1TN,
                                          frameDescr2TN,
                                          frameMatchesTN,
                                          frameHomosTN,
                                          srcImgPatchIdxAndKpTN);
        CV_Assert((frameDescr1TN.rows == frameDescr2TN.rows)
        && (frameDescr1TN.rows == combNrCorrsTN)
        && (frameMatchesTN.size() == (size_t)combNrCorrsTN)
        && (frameHomosTN.size() == (size_t)combNrCorrsTN)
        && (srcImgPatchIdxAndKpTN.size() == (size_t)combNrCorrsTN));
        frameKeypoints1.insert(frameKeypoints1.end(), frameKPsTN1.begin(), frameKPsTN1.end());
        frameKeypoints2.insert(frameKeypoints2.end(), frameKPsTN2.begin(), frameKPsTN2.end());
        frameKeypoints2NoErr.insert(frameKeypoints2NoErr.end(), frameKPsTN2.begin(), frameKPsTN2.end());
        frameDescriptors1.push_back(frameDescr1TN);
        frameDescriptors2.push_back(frameDescr2TN);
        frameMatches.insert(frameMatches.end(), frameMatchesTN.begin(), frameMatchesTN.end());
        frameInliers.insert(frameInliers.end(), (size_t)combNrCorrsTN, false);
        frameHomographies.insert(frameHomographies.end(), frameHomosTN.begin(), frameHomosTN.end());
        vector<Mat> frameHomosTNCam1 = vector<Mat>(combNrCorrsTN, Mat::eye(3, 3, CV_64FC1));
        frameHomographiesCam1.insert(frameHomographiesCam1.end(), frameHomosTNCam1.begin(), frameHomosTNCam1.end());
        srcImgPatchIdxAndKp.insert(srcImgPatchIdxAndKp.end(), srcImgPatchIdxAndKpTN.begin(), srcImgPatchIdxAndKpTN.end());
        CV_Assert(combNrCorrsTN == (finalNrTNStatCorrs + finalNrTNMovCorrs));
        if(combCorrsImg12TNstatFirst){
            corrType.insert(corrType.end(), finalNrTNStatCorrs, 4);
            corrType.insert(corrType.end(), finalNrTNMovCorrs, 5);
        }else{
            corrType.insert(corrType.end(), finalNrTNMovCorrs, 5);
            corrType.insert(corrType.end(), finalNrTNStatCorrs, 4);
        }
    }

    idxs_match23D1.clear();
    idxs_match23D2.clear();
    reOrderSortMatches(frameMatches,
                       frameDescriptors1,
                       frameDescriptors2,
                       frameKeypoints1,
                       frameKeypoints2,
                       frameKeypoints2NoErr,
                       frameInliers,
                       frameHomographies,
                       srcImgPatchIdxAndKp,
                       corrType,
                       frameHomographiesCam1,
                       idxs_match23D1,
                       idxs_match23D2);

    //Write matches to disk
    if(!writeMatchesToDisk()){
        return false;
    }

    featureIdxBegin += nrCorrs[actFrameCnt];

    return true;
}

void genMatchSequ::getErrorFreeKeypoints(const std::vector<cv::KeyPoint> &kpWithErr,
                           std::vector<cv::KeyPoint> &kpNoErr){
    kpNoErr = kpWithErr;
    for (int i = 0; i < combNrCorrsTP; ++i) {
        kpNoErr[i].pt.x = (float) combCorrsImg2TP.at<double>(0, i);
        kpNoErr[i].pt.y = (float) combCorrsImg2TP.at<double>(1, i);
    }
}

bool genMatchSequ::writeMatchesToDisk(){
    if(!checkPathExists(matchDataPath)){
        cerr << "Given path " << matchDataPath << " to store matches does not exist!" << endl;
        return false;
    }

    string singleFrameDataFName = matchSingleFrameBaseFName + "_" + std::to_string(actFrameCnt);
    singleFrameDataFName = genSequFileExtension(singleFrameDataFName);
    singleFrameDataFName = concatPath(matchDataPath, singleFrameDataFName);
    if(!checkOverwriteDelFiles(singleFrameDataFName,
                               "Output file for matching data already exists:", overwriteMatchingFiles)){
        return false;
    } else if(checkFileExists(singleFrameDataFName)){
        cerr << "Unable to write matches to disk. Aborting." << endl;
        return false;
    }

    FileStorage fs = FileStorage(singleFrameDataFName, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << singleFrameDataFName << endl;
        return false;
    }

    fs.writeComment("This file contains matches and additional information for a single frame\n\n");


    fs.writeComment("Keypoints for the first (left or top) stereo cam (there is no 1:1 correspondence between "
                        "frameKeypoints1 and frameKeypoints2 as they are shuffled but the keypoint order of each "
                        "of them is the same compared to their corresponding descriptor Mat (rows))");
    fs << "frameKeypoints1" << frameKeypoints1;

    fs.writeComment("Keypoints for the second (right or bottom) stereo cam (there is no 1:1 correspondence between "
                        "frameKeypoints1 and frameKeypoints2 as they are shuffled but the keypoint order of each "
                        "of them is the same compared to their corresponding descriptor Mat (rows))");
    fs << "frameKeypoints2" << frameKeypoints2;

    fs.writeComment("Descriptors for first (left or top) stereo cam (there is no 1:1 correspondence between "
                        "frameDescriptors1 and frameDescriptors2 as they are shuffled but the descriptor order "
                        "is the same compared to its corresponding keypoint vector frameKeypoints1). "
                        "Descriptors corresponding to the same static 3D point (not for moving objects) in different "
                        "stereo frames are equal");
    fs << "frameDescriptors1" << frameDescriptors1;

    fs.writeComment("Descriptors for second (right or bottom) stereo cam (there is no 1:1 correspondence between "
                        "frameDescriptors1 and frameDescriptors2 as they are shuffled but the descriptor order "
                        "the same compared to its corresponding keypoint vector frameKeypoints2). "
                        "Descriptors corresponding to the same static 3D point (not for moving objects) in different "
                        "stereo frames are similar");
    fs << "frameDescriptors2" << frameDescriptors2;

    fs.writeComment("Matches between features of a single stereo frame. They are sorted based on the descriptor "
                        "distance (smallest first)");
    fs << "frameMatches" << frameMatches;

    fs.writeComment("Indicates if a feature (frameKeypoints1 and corresponding frameDescriptors1) "
                        "is an inlier.");
    fs << "frameInliers" << "[";
    for (auto i : frameInliers) {
        fs << i;
    }
    fs << "]";

    fs.writeComment("Keypoints in the second stereo image without a positioning error (in general, keypoints "
                        "in the first stereo image are without errors)");
    fs << "frameKeypoints2NoErr" << frameKeypoints2NoErr;

    fs.writeComment("Holds the homographies for all patches arround keypoints for warping the patch which is "
                        "then used to calculate the matching descriptor. Homographies corresponding to the same "
                        "static 3D point in different stereo frames are similar");
    fs << "frameHomographies" << "[";
    for (auto &i : frameHomographies) {
        fs << i;
    }
    fs << "]";

    fs.writeComment("Holds homographies for all patches arround keypoints in the first camera (for tracked features) "
                    "for warping the patch which is then used to calculate the matching descriptor. "
                    "Homographies corresponding to the same static 3D point in different stereo frames are similar");
    fs << "frameHomographiesCam1" << "[";
    for (auto &i : frameHomographiesCam1) {
        fs << i;
    }
    fs << "]";

    fs.writeComment("Holds the keypoints from the images used to extract patches for the first keypoint of a match "
                        "(image indices for keypoints are stored in srcImgPatchKpImgIdx1)");
    vector<KeyPoint> origKps;
    origKps.reserve(srcImgPatchIdxAndKp.size());
    for (auto &i : srcImgPatchIdxAndKp){
        origKps.push_back(i.first.second);
    }
    fs << "srcImgPatchKp1" << origKps;
    fs.writeComment("Holds the image indices of the images used to extract patches for every keypoint in "
                        "srcImgPatchKp1 (same order)");
    fs << "srcImgPatchKpImgIdx1" << "[";
    for (auto &i : srcImgPatchIdxAndKp) {
        fs << (int)i.first.first;
    }
    fs << "]";
    fs.writeComment("Holds the keypoints from the images used to extract patches for the second keypoint of a match. "
                    "(image indices for keypoints are stored in srcImgPatchKpImgIdx2)");
    size_t idx = 0;
    for (auto &i : srcImgPatchIdxAndKp){
        origKps[idx++] = i.second.second;
    }
    fs << "srcImgPatchKp2" << origKps;
    fs.writeComment("Holds the image indices of the images used to extract patches for every keypoint in "
                    "srcImgPatchKp2 (same order)");
    fs << "srcImgPatchKpImgIdx2" << "[";
    for (auto &i : srcImgPatchIdxAndKp) {
        fs << (int)i.second.first;
    }
    fs << "]";

    fs.writeComment("Specifies the type of a correspondence (TN from static (=4) or TN from moving (=5) object, "
                        "or TP from a new static (=0), a new moving (=1), an old static (=2), or an old moving (=3) "
                        "object (old means, that the corresponding 3D point emerged before this stereo frame and "
                        "also has one or more correspondences in a different stereo frame))");
    fs << "corrType" << "[";
    for (auto &i : corrType) {
        fs << i;
    }
    fs << "]";

    fs.writeComment("Indices for final correspondences to point from reordered matches and parameters "
                    "like frameKeypoints1, frameDescriptors1, frameHomographies, ... to corresponding "
                    "3D information like combCorrsImg1TP, combCorrsImg12TP_IdxWorld2, ...");
    fs << "idxs_match23D1" << "[";
    for (auto &i : idxs_match23D1) {
        fs << static_cast<int>(i);
    }
    fs << "]";

    fs.writeComment("Indices for final correspondences to point from reordered matches and parameters "
                    "like frameKeypoints2, frameDescriptors2, ... to corresponding 3D information "
                    "like combCorrsImg2TP, ...");
    fs << "idxs_match23D2" << "[";
    for (auto &i : idxs_match23D2) {
        fs << static_cast<int>(i);
    }
    fs << "]";
    fs.release();

    return true;
}

void reOrderSortMatches(std::vector<cv::DMatch> &matches,
                        cv::Mat &descriptor1,
                        cv::Mat &descriptor2,
                        std::vector<cv::KeyPoint> &kp1,
                        std::vector<cv::KeyPoint> &kp2,
                        std::vector<cv::KeyPoint> &kp2NoError,
                        std::vector<bool> &inliers,
                        std::vector<cv::Mat> &homos,
                        std::vector<std::pair<std::pair<size_t,cv::KeyPoint>, std::pair<size_t,cv::KeyPoint>>> &srcImgIdxAndKp,
                        std::vector<int> &corrType,
                        std::vector<cv::Mat> &homosCam1,
                        std::vector<size_t> &idxs1,
                        std::vector<size_t> &idxs2){
    CV_Assert((descriptor1.rows == descriptor2.rows)
    && (descriptor1.rows == (int)kp1.size())
    && (kp1.size() == kp2.size())
    && (kp1.size() == matches.size()));

    //Shuffle descriptors and keypoints of img1
    sort(matches.begin(),
         matches.end(),
         [](DMatch &first, DMatch &second){return first.queryIdx < second.queryIdx;});
//    std::vector<size_t> idxs1, idxs2;
    shuffleVector(idxs1, kp1.size());
    cv::Mat descriptor1_tmp;
    descriptor1_tmp.reserve((size_t)descriptor1.rows);
    for(size_t i = 0; i < idxs1.size(); ++i){
        descriptor1_tmp.push_back(descriptor1.row((int)idxs1[i]));
        matches[idxs1[i]].queryIdx = (int)i;
    }
    descriptor1_tmp.copyTo(descriptor1);
    reOrderVector(kp1, idxs1);
    reOrderVector(inliers, idxs1);
    reOrderVector(homos, idxs1);
    reOrderVector(homosCam1, idxs1);
    reOrderVector(srcImgIdxAndKp, idxs1);
    reOrderVector(corrType, idxs1);

    //Shuffle descriptors and keypoints of img2
    sort(matches.begin(),
         matches.end(),
         [](DMatch &first, DMatch &second){return first.trainIdx < second.trainIdx;});
    shuffleVector(idxs2, kp2.size());
    cv::Mat descriptor2_tmp;
    descriptor2_tmp.reserve((size_t)descriptor2.rows);
    for(size_t i = 0; i < idxs2.size(); ++i){
        descriptor2_tmp.push_back(descriptor2.row((int)idxs2[i]));
        matches[idxs2[i]].trainIdx = (int)i;
    }
    descriptor2_tmp.copyTo(descriptor2);
    reOrderVector(kp2, idxs2);
    reOrderVector(kp2NoError, idxs2);

    //Sort matches based on descriptor distance
    sort(matches.begin(),
         matches.end(),
         [](DMatch &first, DMatch &second){return first.distance < second.distance;});
}

template<typename T, typename T1>
void reOrderVector(std::vector<T> &reOrderVec, std::vector<T1> &idxs){
    CV_Assert(reOrderVec.size() == idxs.size());

    std::vector<T> reOrderVec_tmp;
    reOrderVec_tmp.reserve(reOrderVec.size());
    for(auto& i : idxs){
        reOrderVec_tmp.push_back(reOrderVec[i]);
    }
    reOrderVec = std::move(reOrderVec_tmp);
}

template<typename T>
void shuffleVector(std::vector<T> &idxs, size_t si){
    idxs = vector<T>(si);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::shuffle(idxs.begin(), idxs.end(), std::mt19937{std::random_device{}()});
}

void genMatchSequ::generateCorrespondingFeaturesTPTN(size_t featureIdxBegin_,
                                                     bool useTN,
                                                     std::vector<cv::KeyPoint> &frameKPs1,
                                                     std::vector<cv::KeyPoint> &frameKPs2,
                                                     cv::Mat &frameDescr1,
                                                     cv::Mat &frameDescr2,
                                                     std::vector<cv::DMatch> &frameMatchesTNTP,
                                                     std::vector<cv::Mat> &homo,
                                                     std::vector<std::pair<std::pair<size_t,cv::KeyPoint>,
                                                             std::pair<size_t,cv::KeyPoint>>> &srcImgIdxAndKp,
                                                     std::vector<cv::Mat> *homoCam1){
    //Generate feature for every TP or TN
//    int show_cnt = 0;
//    const int show_interval = 50;
    size_t featureIdx = featureIdxBegin_;
    size_t initCorrID = tntpindexer.getCorrID(featureIdxRepPatt[featureIdx], useTN);
    KeypointIndexer &initkp = corrToIdxMap.at(initCorrID);

    //Calculate image intensity noise distribution for TNs
    double stdNoiseTN = getRandDoubleValRng(2.0, 10.0);
    double meanIntTNNoise = getRandDoubleValRng(0, 2.5 * stdNoiseTN);
    meanIntTNNoise *= pow(-1.0, (double)(rand2() % 2));

    //Check if we need to calculate a keypoint position in the warped patch
    bool kpCalcNeeded = !nearZero((double)initkp.getKeypoint1().angle + 1.0)
            || (initkp.getKeypoint1().octave != 0)
            || (initkp.getKeypoint1().class_id != -1);

    //Maximum descriptor distance for TP
    double ThTp = badDescrTH.median + (badDescrTH.maxVal - badDescrTH.median) / 3.0;
    //Minimum descriptor distance for TP for the fall-back solution
    const double minDescrDistTP = max(min(badDescrTH.median / 8.0, badDescrTH.minVal / 2.0), badDescrTH.minVal / 4.0);
    //Minimum descriptor distance for TN
    double ThTn = min(badDescrTH.mean - 4.0 * badDescrTH.standardDev, 2.0 * badDescrTH.minVal / 3.0);
    ThTn = (ThTn < badDescrTH.minVal / 3.0) ? (2.0 * badDescrTH.minVal / 3.0):ThTn;
    //Minimum descriptor distance for TN which are near to their correct position
    double ThTnNear = (minDescrDistTP + ThTn) / 2.0;

    PatchCInfo patchInfos(minDescrDistTP, useTN, stdNoiseTN, meanIntTNNoise, kpCalcNeeded, ThTp, ThTn, ThTnNear);

    //std::normal_distribution<double> distr;
    int nrcombCorrs;
    if(useTN){
        nrcombCorrs = combNrCorrsTN;
        double posMeanErr = getRandDoubleValRng(0, 5.0);
        double maxErr = getRandDoubleValRng(posMeanErr + 1.0, 10.0);
        double posStdErr = getRandDoubleValRng(0, (maxErr - posMeanErr) / 3.5);
        patchInfos.distr = std::normal_distribution<double>(posMeanErr, posStdErr);
    }else{
        nrcombCorrs = combNrCorrsTP;
        patchInfos.distr = std::normal_distribution<double>(parsMtch.keypErrDistr.first, parsMtch.keypErrDistr.second);
    }

    int gtmcnt = 0, warpcnt = 0;
    int showIntervalWarp = INT_MAX;
    int showIntervalGTM = INT_MAX;
    if(useTN){
        showIntervalWarp = max(static_cast<int>(round(static_cast<double>(nrTNFullSequWarped) *
                                                      static_cast<double>(patchInfos.show_interval) /
                                                      static_cast<double>(nrTNFullSequ))), 1);
    }else{
        showIntervalWarp = max(static_cast<int>(round(static_cast<double>(nrTPFullSequWarped) *
                                                      static_cast<double>(patchInfos.show_interval) /
                                                      static_cast<double>(nrTPFullSequ))), 1);
    }
    showIntervalGTM = patchInfos.show_interval - showIntervalWarp;
    for (int i = 0; i < nrcombCorrs; ++i) {
        patchInfos.featureIdx_tmp = featureIdx;
        patchInfos.i = i;
        //Check if the feature index is higher than the available nr of features
        if (patchInfos.featureIdx_tmp >= tntpindexer.size()){
            //Set the index to a random number in the allowed range
            cerr << "Feature index out of range for a few correspondences. Using new index which "
                    "might point to an already used feature." << endl;
            if(useTN) {
                patchInfos.featureIdx_tmp = tntpindexer.getNextTNID(numeric_limits<size_t>::max(), featureIdxBegin_);
            }else{
                patchInfos.featureIdx_tmp = tntpindexer.getNextTPID(numeric_limits<size_t>::max(), featureIdxBegin_);
            }
        }
        patchInfos.visualize = false;
        if((verbose & SHOW_PLANES_FOR_HOMOGRAPHY) && ((patchInfos.show_cnt % patchInfos.show_interval) == 0)){
            patchInfos.visualize = true;
        }
        patchInfos.show_cnt++;
        //Calculate homography
        Mat H, H1_dist = Mat::eye(3, 3, CV_64FC1);
        bool succ = true;
        bool succCam1 = true;
        if(useTN){
            H = getHomographyForDistortionTN(combCorrsImg1TN.col(i), patchInfos.visualize);
        }else {
            Mat X = Mat(comb3DPts[i], true).reshape(1);
            int64_t idx3D;
            if (combCorrsImg12TP_IdxWorld2.empty()){
                idx3D = combCorrsImg12TP_IdxWorld[i];
            }else{
                idx3D = combCorrsImg12TP_IdxWorld2[i];
            }
            try {
                H = getHomographyForDistortionChkOld(X,
                                                     combCorrsImg1TP.col(i),
                                                     combCorrsImg2TP.col(i),
                                                     idx3D,
                                                     combCorrsImg12TP_IdxWorld[i],
                                                     featureIdx,
                                                     patchInfos.visualize);
            }catch(SequenceException &e){
                cout << "Exception: " << e.what() << endl;
                cout << "Using random homography." << endl;
                succ = false;
                succCam1 = false;
            }
            if(parsMtch.distortPatchCam1 && succ){
                try {
                    H1_dist = getHomographyForDistortionChkOld(X,
                                                               combCorrsImg1TP.col(i),
                                                               combCorrsImg1TP.col(i),
                                                               idx3D,
                                                               combCorrsImg12TP_IdxWorld[i],
                                                               featureIdx,
                                                               patchInfos.visualize,
                                                               true);
                }catch(SequenceException &e){
                    cout << "Exception: " << e.what() << endl;
                    cout << "Using random homography for distorting patches in first camera." << endl;
                    succCam1 = false;
                }
            }
            if(((idx3D >= 0) || !combCorrsImg12TP_IdxWorld2.empty()) && !planeTo3DIdx.empty()){
                if(planeTo3DIdx.find(idx3D) != planeTo3DIdx.end()) {
                    patchInfos.featureIdx_tmp = get<1>(planeTo3DIdx[idx3D]);
                }
            }
        }

        //Get image (check if already in memory)
        Mat img1, img2;
        size_t corr_id = tntpindexer.getCorrID(featureIdxRepPatt[patchInfos.featureIdx_tmp], useTN);
        KeypointIndexer &kpinfo = corrToIdxMap.at(corr_id);
        size_t uimgID1 = kpinfo.getUniqueImgID1();
        size_t uimgID2 = kpinfo.getUniqueImgID2();
        if(loadImgsEveryFrame){
            if(imgFrameIdxMap[actFrameCnt].first.find(uimgID1) != imgFrameIdxMap[actFrameCnt].first.end()){
                img1 = imgs[imgFrameIdxMap[actFrameCnt].first[uimgID1]];
            }else{
                img1 = cv::imread(*uniqueImgIDToName.at(uimgID1), IMREAD_GRAYSCALE);
            }
//            if(*uniqueImgIDToName.at(uimgID1) != kpinfo.getImgName1()){
//                cerr << "Image IDs (img1) not consistend when loading images!" << endl;
//                throw SequenceException("Image IDs not consistend!");
//            }
            if(kpinfo.has2Imgs()){
                if(imgFrameIdxMap[actFrameCnt].first.find(uimgID2) != imgFrameIdxMap[actFrameCnt].first.end()){
                    img2 = imgs[imgFrameIdxMap[actFrameCnt].first[uimgID2]];
                }else{
                    img2 = cv::imread(*uniqueImgIDToName.at(uimgID2), IMREAD_GRAYSCALE);
                }
//                if(*uniqueImgIDToName.at(uimgID2) != kpinfo.getImgName2()){
//                    cerr << "Image IDs (img2) not consistend when loading images!" << endl;
//                    throw SequenceException("Image IDs not consistend!");
//                }
            }else{
                img2 = img1;
            }
        } else{
            img1 = imgs[imgFrameIdxMap[actFrameCnt].first[uimgID1]];
            if(kpinfo.has2Imgs()){
                img2 = imgs[imgFrameIdxMap[actFrameCnt].first[uimgID2]];
            }else{
                img2 = img1;
            }
        }

        //Extract image patch
        KeyPoint kp1 = kpinfo.getKeypoint1();
        KeyPoint kp2;
        if(kpinfo.has2Imgs()) {
            kp2 = kpinfo.getKeypoint2();
        }else{
            kp2 = kp1;
        }
        srcImgIdxAndKp.emplace_back(make_pair(uimgID1, kp1), make_pair(uimgID2, kp2));

        cv::KeyPoint kp21;
        cv::Point2f kp2err1 = cv::Point2f(0, 0);
        double descrDist1;
        cv::Mat descr11;
        bool c1_distort = false;
        if(!isIdentityMat(H1_dist)){
            patchInfos.succ = succCam1;
            patchInfos.takeImg2FallBack = false;
            descr11 = calculateDescriptorWarped(img1, kp1, H1_dist, *homoCam1, patchInfos, kp21, kp2err1, descrDist1, true);
            if(succ) {
                H = H1_dist * H;
            }
            c1_distort = true;
        }else if(!useTN && (homoCam1 != nullptr)){
            homoCam1->emplace_back(Mat::eye(3, 3, CV_64FC1));
        }

        cv::KeyPoint kp3;
        cv::Point2f kp2err = cv::Point2f(0, 0);
        double descrDist;
        patchInfos.succ = succ;
        patchInfos.takeImg2FallBack = kpinfo.has2Imgs();
        cv::Mat descr21;
        if(!c1_distort && kpinfo.has2Imgs()){
            homo.emplace_back(Mat::eye(3, 3, CV_64FC1));
            kp3 = kp2;
            if(useTN || (nearZero(parsMtch.keypErrDistr.first) && nearZero(parsMtch.keypErrDistr.second))) {
                descr21 = kpinfo.getDescriptor2();
                descrDist = kpinfo.getDescriptorDist();
            }else {
                cv::Rect roi2 = getNormalPatchROI(img2.size(), kp3);
                cv::KeyPoint kp4 = kp3;
                kp4.pt.x -= static_cast<float>(roi2.x);
                kp4.pt.y -= static_cast<float>(roi2.y);
                distortKeyPointPosition(kp4, roi2, patchInfos.distr);
                vector<KeyPoint> kpv_tmp(1, kp4);
                Mat img2_roi = img2(roi2);
                if (matchinglib::getDescriptors(img2_roi,
                                                kpv_tmp,
                                                parsMtch.descriptorType,
                                                descr21,
                                                parsMtch.keyPointType) != 0) {
                    kp3 = kp2;
                    descr21 = kpinfo.getDescriptor2();
                }else{
                    kp4.pt.x += static_cast<float>(roi2.x);
                    kp4.pt.y += static_cast<float>(roi2.y);
                    kp3 = kp4;
                }
                descrDist = kpinfo.getDescriptorDist(descr21);
                if (descrDist > ThTp) {
                    kp3 = kp2;
                    descr21 = kpinfo.getDescriptor2();
                    descrDist = kpinfo.getDescriptorDist();
                }
                kp2err = kp3.pt - kp2.pt;
            }
            patchInfos.patchROIimg1 = getNormalPatchROI(img1.size(), kp1);
            patchInfos.patchROIimg2 = getNormalPatchROI(img2.size(), kp3);
        }else {
            descr21 = calculateDescriptorWarped(img2, kp2, H, homo, patchInfos, kp3, kp2err,
                                                descrDist, false, H1_dist, descr11);
        }

//        if(kpinfo.has2Imgs()) {
//            cout << kpinfo.getImgName1() << " --> " << kpinfo.getImgName2() << endl;
//        }else{
//            cout << kpinfo.getImgName1() << " --> " << kpinfo.getImgName1() << endl;
//        }
//        cout << *uniqueImgIDToName.at(uimgID1) << " --> " << *uniqueImgIDToName.at(uimgID2) << endl << endl;

        bool showPatches[6];
        showPatches[0] = (verbose & SHOW_CORRESP_PATCHES_GTM);
        showPatches[1] = kpinfo.isGTM() && showPatches[0];
        showPatches[2] = (verbose & SHOW_CORRESP_PATCHES_WARPED);
        showPatches[3] = !kpinfo.isGTM() && showPatches[2];
        showPatches[4] = (verbose & SHOW_CORRESP_PATCHES_TN);
        showPatches[5] = useTN && showPatches[4];
        bool showit = showPatches[5] && !showPatches[0] && !showPatches[2] &&
                (((patchInfos.show_cnt - 1) % patchInfos.show_interval) == 0);
        showit |= showPatches[5] && showPatches[1] && ((gtmcnt % showIntervalGTM) == 0);
        showit |= showPatches[5] && showPatches[3] && ((warpcnt % showIntervalWarp) == 0);
        showit |= !useTN && (!showit ^ showPatches[4]) && showPatches[1] && ((gtmcnt % showIntervalGTM) == 0);
        showit |= !useTN && (!showit ^ showPatches[4]) && showPatches[3] && ((warpcnt % showIntervalWarp) == 0);
        if(showit) {
            bool show = true;
            if(patchInfos.patch1.empty() || patchInfos.patch2.empty()){
                if(patchInfos.patchROIimg1.x < 0 || patchInfos.patchROIimg1.y < 0 ||
                   patchInfos.patchROIimg1.x + patchInfos.patchROIimg1.width > img1.cols ||
                   patchInfos.patchROIimg1.y + patchInfos.patchROIimg1.height > img1.rows){
                    show = false;
                }
                if(patchInfos.patchROIimg2.x < 0 || patchInfos.patchROIimg2.y < 0 ||
                   patchInfos.patchROIimg2.x + patchInfos.patchROIimg2.width > img2.cols ||
                   patchInfos.patchROIimg2.y + patchInfos.patchROIimg2.height > img2.rows){
                    show = false;
                }
                if(show){
                    img1(patchInfos.patchROIimg1).copyTo(patchInfos.patch1);
                    img2(patchInfos.patchROIimg2).copyTo(patchInfos.patch2);
                }
            }
            if(show) {
                showCorrespPatches(patchInfos.patch1, patchInfos.patch2);
            }
        }
        patchInfos.patch1.release();
        patchInfos.patch2.release();
        patchInfos.patchROIimg1.x = -1;
        patchInfos.patchROIimg2.y = -1;
        if(showPatches[1]){ gtmcnt++; }
        if(showPatches[3]){ warpcnt++; }

        //Store the keypoints and descriptors
        if(useTN){
            kp1.pt.x = (float) combCorrsImg1TN.at<double>(0, i);
            kp1.pt.y = (float) combCorrsImg1TN.at<double>(1, i);
            kp3.pt.x = (float) combCorrsImg2TN.at<double>(0, i);
            kp3.pt.y = (float) combCorrsImg2TN.at<double>(1, i);
        }else {
            if(c1_distort){
                kp1 = kp21;
            }
            kp1.pt.x = (float) combCorrsImg1TP.at<double>(0, i);
            kp1.pt.y = (float) combCorrsImg1TP.at<double>(1, i);
            kp3.pt.x = (float) combCorrsImg2TP.at<double>(0, i) + kp2err.x;
            kp3.pt.y = (float) combCorrsImg2TP.at<double>(1, i) + kp2err.y;
            if(kp3.pt.x > ((float)imgSize.width - 1.f)){
                kp2err.x -= kp3.pt.x - (float)imgSize.width - 1.f;
                kp3.pt.x = (float)imgSize.width - 1.f;
            }else if(kp3.pt.x < 0){
                kp2err.x -= kp3.pt.x;
                kp3.pt.x = 0;
            }
            if(kp3.pt.y > ((float)imgSize.height - 1.f)){
                kp2err.y -= kp3.pt.y - (float)imgSize.height - 1.f;
                kp3.pt.y = (float)imgSize.height - 1.f;
            }else if(kp3.pt.y < 0){
                kp2err.y -= kp3.pt.y;
                kp3.pt.y = 0;
            }
            kpErrors.push_back((double)sqrt(kp2err.x * kp2err.x + kp2err.y * kp2err.y));
        }
        frameKPs1[i] = kp1;
        frameKPs2[i] = kp3;
        if(c1_distort){
            frameDescr1.push_back(descr11.clone());
        }else {
            frameDescr1.push_back(kpinfo.getDescriptor1().clone());
        }
        frameDescr2.push_back(descr21.clone());
        frameMatchesTNTP.emplace_back(DMatch(i, i, (float)descrDist));

        featureIdx++;
    }
}

cv::Rect genMatchSequ::getNormalPatchROI(const cv::Size &imgSi, const cv::KeyPoint &kp) const{
    Point mid(static_cast<int>(round(kp.pt.x)), static_cast<int>(round(kp.pt.y)));
    const int halfSi = (minPatchSize2 - 1) / 2;
    Point tl(mid.x - halfSi, mid.y - halfSi);
    Size pSi(minPatchSize2, minPatchSize2);
    if(tl.x + pSi.width > imgSi.width){
        pSi.width = imgSi.width - tl.x;
    }
    if(tl.y + pSi.height > imgSi.height){
        pSi.height = imgSi.height - tl.y;
    }
    if(tl.x < 0) tl.x = 0;
    if(tl.y < 0) tl.y = 0;
    return {tl, pSi};
}

cv::Mat genMatchSequ::calculateDescriptorWarped(const cv::Mat &img,
                                                const cv::KeyPoint &kp,
                                                cv::Mat &H,
                                                std::vector<cv::Mat> &homo,
                                                PatchCInfo &patchInfos,
                                                cv::KeyPoint &kp2,
                                                cv::Point2f &kp2err,
                                                double &descrDist,
                                                bool forCam1,
                                                cv::InputArray H_cam1,
                                                cv::InputArray descr_cam1){
    //Calculate the rotated ellipse from the keypoint size (circle) after applying the homography to the circle
    // to estimate the minimal necessary patch size
    cv::Rect patchROIimg1(0,0,3,3), patchROIimg2(0,0,3,3), patchROIimg21(0,0,3,3);
    cv::Point2d ellipseCenter;
    double ellipseRot = 0;
    cv::Size2d axes;
    bool useFallBack = false;
    bool noEllipse = false;
    bool reflectionX = false;
    bool reflectionY = false;
    cv::Size imgFeatureSize = img.size();
    const int minImgDim = min(imgFeatureSize.width, imgFeatureSize.height);
    int minPatchSize_tmp = min(41, minImgDim);
    minPatchSize_tmp -= (minPatchSize_tmp + 1) % 2;
    const auto minPatchSize = static_cast<double>(minPatchSize_tmp);
    const size_t corrID = tntpindexer.getCorrID(featureIdxRepPatt[patchInfos.featureIdx_tmp], patchInfos.useTN);
    const bool init_succ = patchInfos.succ;
    if(patchInfos.succ) {
        patchInfos.succ = getRectFitsInEllipse(H,
                                               kp,
                                               patchROIimg1,
                                               patchROIimg2,
                                               patchROIimg21,
                                               ellipseCenter,
                                               ellipseRot,
                                               axes,
                                               reflectionX,
                                               reflectionY,
                                               imgFeatureSize);
    }
    if(!patchInfos.succ){
        //If the calculation of the necessary patch size failed, calculate a standard patch
        noEllipse = true;
        int fbCnt = 0;
        do {
            fbCnt++;
            useFallBack = false;
            int patchSize = minPatchSize2;//Must be an odd number
            if(patchSize > minImgDim){
                patchSize = minImgDim;
                patchSize -= (patchSize + 1) % 2;
            }
            if(!H.empty()) {
                do {
                    Mat kpm = (Mat_<double>(3, 1) << (double) kp.pt.x, (double) kp.pt.y, 1.0);
                    kpm = H * kpm;
                    kpm /= kpm.at<double>(2);
                    Point2i midPt((int) round(kpm.at<double>(0)), (int) round(kpm.at<double>(1)));

                    if (!getImgROIs(H,
                                    midPt,
                                    patchSize,
                                    patchROIimg1,
                                    patchROIimg2,
                                    patchROIimg21,
                                    reflectionX,
                                    reflectionY,
                                    imgFeatureSize,
                                    kp,
                                    maxPatchSizeMult2)) {
                        useFallBack = true;
                        break;
                    }

                    if ((patchROIimg1.width > (double) (maxPatchSizeMult2 * minPatchSize2))
                        || (patchROIimg1.height > (double) (maxPatchSizeMult2 * minPatchSize2))
                        || (patchROIimg2.width > (double) (maxPatchSizeMult2 * minPatchSize2))
                        || (patchROIimg2.height > (double) (maxPatchSizeMult2 * minPatchSize2))) {
                        useFallBack = true;
                        break;
                    }

                    if ((patchROIimg2.width < minPatchSize) ||
                        (patchROIimg2.height < minPatchSize)) {
                        if(patchSize < minImgDim) {
                            //Calc a bigger patch size for the warped patch
                            patchSize = (int) ceil(1.2f * (float) patchSize);
                            patchSize += (patchSize + 1) % 2;//Must be an odd number
                        }
                    }

                } while (((patchROIimg2.width < minPatchSize) || (patchROIimg2.height < minPatchSize)) &&
                         !useFallBack);
            }else{
                useFallBack = true;
            }
            if (useFallBack) {
                //Construct a random affine homography
                //Generate the non-isotropic scaling of the deformation (shear)
                double d1, d2;
                d1 = getRandDoubleValRng(0.8, 1.0);
                d2 = getRandDoubleValRng(0.8, 1.0);
                size_t sign1 = rand2() % 6;
                if (sign1 == 0) {
                    d1 *= -1.0;
                }
                sign1 = rand2() % 6;
                if (sign1 == 0) {
                    d2 *= -1.0;
                }
                Mat D = Mat::eye(2, 2, CV_64FC1);
                D.at<double>(0, 0) = d1;
                D.at<double>(1, 1) = d2;
                //Generate a rotation for the deformation (shear)
                double angle_rot = getRandDoubleValRng(0, M_PI_4);
                Mat Rdeform = (Mat_<double>(2, 2) << std::cos(angle_rot), (-1. * std::sin(angle_rot)),
                        std::sin(angle_rot), std::cos(angle_rot));
                //Generate a rotation
                angle_rot = getRandDoubleValRng(0, M_PI_2);
                Mat Rrot = (Mat_<double>(2, 2) << std::cos(angle_rot), (-1. * std::sin(angle_rot)),
                        std::sin(angle_rot), std::cos(angle_rot));
                double scale = getRandDoubleValRng(0.65, 1.35);
                if(fbCnt > 5){
                    scale *= (double)fbCnt / 3.5;
                }
                //Calculate the new affine homography (without translation)
                Mat Haff2 = scale * Rrot * Rdeform.t() * D * Rdeform;
                H = Mat::eye(3, 3, CV_64FC1);
                Haff2.copyTo(H.colRange(0, 2).rowRange(0, 2));
                if(!forCam1 && !init_succ && !H_cam1.empty()){
                    Mat H_cam1m = H_cam1.getMat();
                    if(!H_cam1m.empty()){
                        H = H_cam1m * H;
                    }
                }
                //Eliminate the translation
//                Mat x1 = (Mat_<double>(3, 1) << (double) kp.pt.x, (double) kp.pt.y, 1.0);
//                Mat tm = H * x1;
//                tm /= tm.at<double>(2);
//                tm = x1 - tm;
//                if(!nearZero(cv::norm(tm))) {
//                    Mat tback = Mat::eye(3, 3, CV_64FC1);
//                    tback.at<double>(0, 2) = tm.at<double>(0);
//                    tback.at<double>(1, 2) = tm.at<double>(1);
//                    H = tback * H;
//                }
            }
        }while(useFallBack && (fbCnt < 21));
    }

    //Extract and warp the patch if the patch size is valid

    //Adapt H to eliminate wrong translation inside the patch
    //Translation to start at (0,0) in the warped image for the selected ROI arround the original image
    // (with coordinates in the original image based on the full image)
    Mat wiTo0 = Mat::eye(3,3,CV_64FC1);
    wiTo0.at<double>(0,2) = -1.0 * (double)patchROIimg21.x;
    wiTo0.at<double>(1,2) = -1.0 * (double)patchROIimg21.y;
    Mat H1 = wiTo0 * H;
    //Translation for the original image to ensure that points starting (left upper corner) at (0,0)
    // are mapped to the image ROI of the warped image

    //Check for reflection
    Mat x2;
    if(reflectionX && reflectionY){
        x2 = (Mat_<double>(3,1) << (double)patchROIimg21.x + (double)patchROIimg21.width - 1.0,
                (double)patchROIimg21.y + (double)patchROIimg21.height - 1.0, 1.0);
    }else if(reflectionX){
        x2 = (Mat_<double>(3,1) << (double)patchROIimg21.x,
                (double)patchROIimg21.y + (double)patchROIimg21.height - 1.0, 1.0);
    }else if(reflectionY){
        x2 = (Mat_<double>(3,1) << (double)patchROIimg21.x + (double)patchROIimg21.width - 1.0,
                (double)patchROIimg21.y, 1.0);
    }else{
        x2 = (Mat_<double>(3,1) << (double)patchROIimg21.x,
                (double)patchROIimg21.y, 1.0);
    }

    Mat Hi = H.inv();
    Mat tm = Hi * x2;
    tm /= tm.at<double>(2);
    Mat tback = Mat::eye(3,3,CV_64FC1);
    tback.at<double>(0,2) = tm.at<double>(0);
    tback.at<double>(1,2) = tm.at<double>(1);
    Mat H2 = H1 * tback;
    Mat patchw;
    if (!useFallBack) {
        warpPerspective(img(patchROIimg1), patchw, H2, patchROIimg21.size(), INTER_LINEAR, BORDER_REPLICATE);
    }
    homo.push_back(H.clone());
    //Extract warped patch ROI with only valid pixels
    if (!useFallBack) {
        patchROIimg2.x = patchROIimg2.x - patchROIimg21.x;
        patchROIimg2.y = patchROIimg2.y - patchROIimg21.y;
        patchw = patchw(patchROIimg2);
    }

    //Adapt center of ellipse
    if(!noEllipse){
        Mat xe = (Mat_<double>(3,1) << ellipseCenter.x,
                ellipseCenter.y, 1.0);
        xe = Hi * xe;
        xe /= xe.at<double>(2);
        xe.at<double>(0) -= (double)patchROIimg1.x;
        xe.at<double>(1) -= (double)patchROIimg1.y;
        xe = H2 * xe;
        xe /= xe.at<double>(2);
        ellipseCenter.x = xe.at<double>(0) - (double)patchROIimg2.x;
        ellipseCenter.y = xe.at<double>(1) - (double)patchROIimg2.y;
        if((ellipseCenter.x < 0) || (ellipseCenter.y < 0)){
            useFallBack = true;
            noEllipse = true;
        }
    }

    //Show the patches
    if(!useFallBack && (verbose & SHOW_WARPED_PATCHES) && (((patchInfos.show_cnt - 1) % patchInfos.show_interval) == 0)){
        //Show the warped patch with/without warped keypoint size (ellipse)
        int border1x = 0, border1y = 0, border2x = 0, border2y = 0;
        if(patchROIimg1.width > patchROIimg2.width){
            border2x = patchROIimg1.width - patchROIimg2.width;
        }else{
            border1x =  patchROIimg2.width - patchROIimg1.width;
        }
        if(patchROIimg1.height > patchROIimg2.height){
            border2y = patchROIimg1.height - patchROIimg2.height;
        }else{
            border1y =  patchROIimg2.height - patchROIimg1.height;
        }
        if(noEllipse){
            Mat patchwc, patchc;
            if(border2x || border2y) {
                cv::copyMakeBorder(patchw, patchwc, 0, border2y, 0, border2x, BORDER_CONSTANT, Scalar::all(0));
            }else{
                patchwc = patchw;
            }
            if(border1x || border1y) {
                cv::copyMakeBorder(img(patchROIimg1), patchc, 0, border1y, 0, border1x, BORDER_CONSTANT, Scalar::all(0));
            }else{
                patchc = img(patchROIimg1);
            }
            Mat bothPathes;
            cv::hconcat(patchc, patchwc, bothPathes);
            namedWindow("Original and warped patch", WINDOW_AUTOSIZE);
            imshow("Original and warped patch", bothPathes);

            waitKey(0);
            destroyWindow("Original and warped patch");
        }else{
            //Transform the ellipse center position
            cv::Point2d ellipseCenter1 = ellipseCenter;
            /*ellipseCenter1.x -= patchROIimg21.x + patchROIimg2.x;
            ellipseCenter1.y -= patchROIimg21.y + patchROIimg2.y;*/
            cv::Point c((int)round(ellipseCenter1.x), (int)round(ellipseCenter1.y));
            CV_Assert((ellipseCenter1.x >= 0) && (ellipseCenter1.y >= 0)
                      && (ellipseCenter1.x < (double)patchROIimg2.width)
                      && (ellipseCenter1.y < (double)patchROIimg2.height));
            cv::Size si((int)round(axes.width), (int)round(axes.height));
            Mat patchwc;
            cvtColor(patchw, patchwc, cv::COLOR_GRAY2BGR);
            cv::ellipse(patchwc, c, si, ellipseRot, 0, 360.0, Scalar(0,0,255));
            //Draw exact correspondence location
            Mat kpm = (Mat_<double>(3,1) << (double)kp.pt.x - (double)patchROIimg1.x,
                    (double)kp.pt.y - (double)patchROIimg1.y, 1.0);
            kpm = H2 * kpm;
            kpm /= kpm.at<double>(2);
            c = Point((int)round(kpm.at<double>(0)) - patchROIimg2.x, (int)round(kpm.at<double>(1)) - patchROIimg2.y);
            cv::circle(patchwc, c, 1, Scalar(0,255,0));
            if(border2x || border2y) {
                cv::copyMakeBorder(patchwc, patchwc, 0, border2y, 0, border2x, BORDER_CONSTANT, Scalar::all(0));
            }
            Mat patchc;
            cvtColor(img(patchROIimg1), patchc, cv::COLOR_GRAY2BGR);
            c = Point((int)round(kp.pt.x) - patchROIimg1.x, (int)round(kp.pt.y) - patchROIimg1.y);
            cv::circle(patchc, c, (int)round(kp.size / 2.f), Scalar(0,0,255));
            //Draw exact correspondence location
            cv::circle(patchc, c, 1, Scalar(0,255,0));
            if(border1x || border1y) {
                cv::copyMakeBorder(patchc, patchc, 0, border1y, 0, border1x, BORDER_CONSTANT, Scalar::all(0));
            }
            Mat bothPathes;
            cv::hconcat(patchc, patchwc, bothPathes);

            //Show correspondence in original image
            /*Mat fullimg;
            cvtColor(img, fullimg, cv::COLOR_GRAY2BGR);
            c = Point((int)round(kp.pt.x), (int)round(kp.pt.y));
            cv::circle(fullimg, c, (int)round(kp.size / 2.f), Scalar(0,0,255));
            //Draw exact correspondence location
            cv::circle(fullimg, c, 1, Scalar(0,255,0));
            namedWindow("Original image with keypoint", WINDOW_AUTOSIZE);
            imshow("Original image with keypoint", fullimg);*/

            namedWindow("Original and warped patch with keypoint", WINDOW_AUTOSIZE);
            imshow("Original and warped patch with keypoint", bothPathes);

            waitKey(0);
            destroyWindow("Original and warped patch with keypoint");
//                destroyWindow("Original image with keypoint");
        }
    }

    //Get the exact position of the keypoint in the patch
    Mat kpm = (Mat_<double>(3,1) << (double)kp.pt.x - (double)patchROIimg1.x,
            (double)kp.pt.y - (double)patchROIimg1.y, 1.0);
    kpm = H2 * kpm;
    kpm /= kpm.at<double>(2);
    kpm.at<double>(0) -= (double)patchROIimg2.x;
    kpm.at<double>(1) -= (double)patchROIimg2.y;
    Point2f ptm = Point2f((float)kpm.at<double>(0), (float)kpm.at<double>(1));

    //Get the difference of the ellipse center and the warped keypoint location
    if(!noEllipse) {
        double diffxKpEx = abs(ellipseCenter.x - kpm.at<double>(0));
        double diffxKpEy = abs(ellipseCenter.y - kpm.at<double>(1));
        double diffxKpEc = sqrt(diffxKpEx * diffxKpEx + diffxKpEy * diffxKpEy);
        if(diffxKpEc > 5.0){
            useFallBack = true;
            noEllipse = true;
        }
    }

    //Check if the used keypoint location is too near to the border of the patch
    const double minPatchSize12 = (double)(minPatchSize - 1) / 2.0;
    const double distKpBx = (double)patchROIimg2.width - kpm.at<double>(0);
    const double distKpBy = (double)patchROIimg2.height - kpm.at<double>(1);
    if((kpm.at<double>(0) < minPatchSize12)
       || (kpm.at<double>(1) < minPatchSize12)
       || (distKpBx < minPatchSize12)
       || (distKpBy < minPatchSize12)){
        useFallBack = true;
        noEllipse = true;
    }

    //Check if we have to use a keypoint detector
    bool keypDetNeed = (noEllipse || patchInfos.kpCalcNeeded);
//    cv::KeyPoint kp2;
//    Point2f kp2err;
    if(!useFallBack && ((!patchInfos.useTN && parsMtch.keypPosErrType && !forCam1) || keypDetNeed)){
        vector<KeyPoint> kps2;
        if (matchinglib::getKeypoints(patchw, kps2, parsMtch.keyPointType, false) != 0) {
            if(kps2.empty()){
                useFallBack = true;
            }
        }
        if(!useFallBack){
            vector<pair<size_t,float>> dists(kps2.size());
            for (size_t j = 0; j < kps2.size(); ++j) {
                float diffx = kps2[j].pt.x - ptm.x;
                float diffy = kps2[j].pt.y - ptm.y;
                dists[j] = make_pair(j, sqrt(diffx * diffx + diffy * diffy));
            }
            sort(dists.begin(), dists.end(),
                 [](pair<size_t,float> &first, pair<size_t,float> &second){return first.second < second.second;});
            if(dists[0].second > 5.f){
                useFallBack = true;
            }else if(noEllipse){
                kp2 = kps2[dists[0].first];
                kp2err.x = kp2.pt.x - ptm.x;
                kp2err.y = kp2.pt.y - ptm.y;
            } else{
                size_t j = 0;
                for (; j < dists.size(); ++j) {
                    if(dists[j].second > 5.f){
                        break;
                    }
                }
                if(j > 1) {
                    //Calculate the overlap area between the found keypoints and the ellipse
                    cv::Point2d ellipseCenter1 = ellipseCenter;
                    /*ellipseCenter1.x -= patchROIimg21.x + patchROIimg2.x;
                    ellipseCenter1.y -= patchROIimg21.y + patchROIimg2.y;*/
                    cv::Point c((int) round(ellipseCenter1.x), (int) round(ellipseCenter1.y));
                    CV_Assert((ellipseCenter1.x >= 0) && (ellipseCenter1.y >= 0)
                              && (ellipseCenter1.x < (double) patchROIimg2.width)
                              && (ellipseCenter1.y < (double) patchROIimg2.height));
                    cv::Size si((int) round(axes.width), (int) round(axes.height));
                    Mat patchmask = Mat::zeros(patchw.size(), patchw.type());
                    cv::ellipse(patchmask, c, si, ellipseRot, 0, 360.0, Scalar::all(255), -1);
                    vector<pair<size_t, int>> overlapareas(j);
                    for (size_t k = 0; k < j; ++k) {
                        Mat patchmask1 = Mat::zeros(patchw.size(), patchw.type());
                        Point c1 = Point((int) round(kps2[dists[k].first].pt.x),
                                         (int) round(kps2[dists[k].first].pt.y));
                        cv::circle(patchmask1, c1, (int) round(kps2[dists[k].first].size / 2.f), Scalar::all(255),
                                   -1);
                        Mat resmask = patchmask & patchmask1;
                        int ovlap = cv::countNonZero(resmask);
                        overlapareas[k] = make_pair(dists[k].first, ovlap);
                    }
                    sort(overlapareas.begin(), overlapareas.end(),
                         [](pair<size_t,int> &first, pair<size_t,int> &second){return first.second > second.second;});
                    //Take the highest overlap areas that are equal
                    int k = 1;
                    for(;k < (int)j; k++){
                        if(overlapareas[0].second != overlapareas[k].second){
                            break;
                        }
                    }
                    if(k > 1){
                        //Get the keypoint with the smallest distance to the exact location
                        vector<size_t> kpIdxSm;
                        for (int l = 0; l < k; ++l) {
                            for (size_t m = 0; m < j; ++m) {
                                if(overlapareas[l].first == dists[m].first){
                                    kpIdxSm.push_back(m);
                                    break;
                                }
                            }
                        }
                        size_t kpSingleIdx = *min_element(kpIdxSm.begin(), kpIdxSm.end());
                        //Take the keypoint with the smallest distance and largest overlap
                        kp2 = kps2[dists[kpSingleIdx].first];
                    }
                    else{
                        //Take the keypoint with the largest overlap
                        kp2 = kps2[overlapareas[0].first];
                    }
                }else{
                    kp2 = kps2[dists[0].first];
                }
                kp2err.x = kp2.pt.x - ptm.x;
                kp2err.y = kp2.pt.y - ptm.y;
            }
            if((!parsMtch.keypPosErrType || patchInfos.useTN) && !useFallBack){
                //Correct the keypoint position to the exact location
                kp2.pt = ptm;
                //Change to keypoint position based on the given error range
                if(!forCam1) {
                    distortKeyPointPosition(kp2, patchROIimg2, patchInfos.distr);
                }
                kp2err.x = kp2.pt.x - ptm.x;
                kp2err.y = kp2.pt.y - ptm.y;
            }
        }
    } else if(!noEllipse){
        //Use the dimension of the ellipse to get the scale/size of the keypoint
        kp2 = kp;
        kp2.pt = ptm;
        kp2.size = 2.f * (float)axes.width;
        //Change to keypoint position based on the given error range
        if(!forCam1) {
            distortKeyPointPosition(kp2, patchROIimg2, patchInfos.distr);
        }
        kp2err.x = kp2.pt.x - ptm.x;
        kp2err.y = kp2.pt.y - ptm.y;
    }

    //Calculate the descriptor
    Mat patchwn;
    Mat descr21;
    descrDist = -1.0;
    bool visPatchNoise = false;
    if((verbose & SHOW_PATCHES_WITH_NOISE) && (((patchInfos.show_cnt - 1) % patchInfos.show_interval) == 0)){
        visPatchNoise = true;
    }
    if(!useFallBack){
        //Apply noise
        if(patchInfos.useTN && !patchInfos.takeImg2FallBack){
            Mat patchwnsp;
            if(combDistTNtoReal[patchInfos.i] < 10.0){
                double stdNoiseTNNear = 2.0 * max(2.0, patchInfos.stdNoiseTN / (1.0 + (10.0 - combDistTNtoReal[patchInfos.i]) / 10.0));
                double meanIntTNNoiseNear = patchInfos.meanIntTNNoise / (1.0 + (10.0 - combDistTNtoReal[patchInfos.i]) / 10.0);
                addImgNoiseGauss(patchw, patchwnsp, meanIntTNNoiseNear, stdNoiseTNNear,
                                 visPatchNoise);
                if(combDistTNtoReal[patchInfos.i] > 5.0) {
                    addImgNoiseSaltAndPepper(patchwnsp, patchwn, 28, 227, visPatchNoise);
                }else{
                    patchwnsp.copyTo(patchwn);
                }
            }else {
                addImgNoiseGauss(patchw, patchwnsp, patchInfos.meanIntTNNoise, 2.0 * patchInfos.stdNoiseTN,
                                 visPatchNoise);
                addImgNoiseSaltAndPepper(patchwnsp, patchwn, 32, 223, visPatchNoise);
            }
        }else if(!patchInfos.takeImg2FallBack){
            if (!nearZero(parsMtch.imgIntNoise.first) || !nearZero(parsMtch.imgIntNoise.second)) {
                addImgNoiseGauss(patchw, patchwn, parsMtch.imgIntNoise.first, parsMtch.imgIntNoise.second,
                                 visPatchNoise);
            }
        }else{
            patchw.copyTo(patchwn);
        }

        //Get descriptor
        vector<KeyPoint> pkp21(1, kp2);
//            pkp21[0] = kp2;
        if (matchinglib::getDescriptors(patchwn,
                                        pkp21,
                                        parsMtch.descriptorType,
                                        descr21,
                                        parsMtch.keyPointType) != 0) {
            useFallBack = true;
        }else{
            //Check matchability
            if(!forCam1 && !descr_cam1.empty()){
                Mat descr1 = descr_cam1.getMat();
                if(!descr1.empty()){
                    descrDist = getDescriptorDistance(descr1, descr21);
                }else{
                    descrDist = getDescriptorDistance(corrToIdxMap.at(corrID).getDescriptor1(), descr21);
                }
            }else {
                descrDist = getDescriptorDistance(corrToIdxMap.at(corrID).getDescriptor1(), descr21);
            }
            if(patchInfos.useTN){
                if (!patchInfos.takeImg2FallBack &&
                (((combDistTNtoReal[patchInfos.i] >= 10.0) && (descrDist < patchInfos.ThTn)) ||
                (descrDist < patchInfos.ThTnNear))) {
                    useFallBack = true;
                }
            }else {
                if (parsMtch.checkDescriptorDist &&
                    ((!patchInfos.takeImg2FallBack && (descrDist > patchInfos.ThTp)) ||
                     (patchInfos.takeImg2FallBack && (descrDist > 1.2 * patchInfos.ThTp)))) {
                    useFallBack = true;
                }
            }
        }
    }

    if(useFallBack){
        //Only add gaussian noise and salt and pepper noise to the original patch
        if(!forCam1) {
            Mat H_cam1m;
            if(!H_cam1.empty()){
                H_cam1m = H_cam1.getMat();
            }
            if(H_cam1m.empty()) {
                homo.back() = Mat::eye(3, 3, CV_64FC1);
            }else{
                H_cam1m.copyTo(homo.back());
            }
        }else{
            homo.back() = Mat::eye(3, 3, CV_64FC1);
        }
        double meang, stdg;
        meang = getRandDoubleValRng(-10.0, 10.0);
        stdg = getRandDoubleValRng(-10.0, 10.0);
        bool fullImgUsed = false;
        Mat patchfb;
        int minPatchSize2_img = min(minPatchSize2, minImgDim);
        minPatchSize2_img -= (minPatchSize2_img + 1) % 2;
        Point2i kp_ri = Point2i((int)round(kp.pt.x), (int)round(kp.pt.y));
        if((patchROIimg1.width < minPatchSize) ||
           ((patchROIimg1.x + patchROIimg1.width) > img.size().width) ||
           (patchROIimg1.height < minPatchSize) ||
           ((patchROIimg1.y + patchROIimg1.height) > img.size().height) ||
           (patchROIimg1.x < 0) ||
           (patchROIimg1.y < 0) ||
           (kp_ri.x < (patchROIimg1.x + 10)) ||
           (kp_ri.x > (patchROIimg1.x + patchROIimg1.width - 10)) ||
           (kp_ri.y < (patchROIimg1.y + 10)) ||
           (kp_ri.y > (patchROIimg1.y + patchROIimg1.height - 10))){
            int ps21 = (minPatchSize2_img - 1) / 2;
            patchROIimg1 = Rect((int)round(kp.pt.x) - ps21,
                                (int)round(kp.pt.y) - ps21,
                                minPatchSize2_img,
                                minPatchSize2_img);
            if(patchROIimg1.x < 0){
                patchROIimg1.width -= patchROIimg1.x;
                patchROIimg1.x = 0;
                if(patchROIimg1.width > img.size().width){
                    patchROIimg1.width = img.size().width;
                }
            }else if((patchROIimg1.x + patchROIimg1.width) > img.size().width){
                int wdiff = patchROIimg1.x + patchROIimg1.width - img.size().width;
                patchROIimg1.x -= wdiff;
                if(patchROIimg1.x < 0){
                    patchROIimg1.x = 0;
                    patchROIimg1.width = img.size().width;
                }
            }
            if(patchROIimg1.y < 0){
                patchROIimg1.height -= patchROIimg1.y;
                patchROIimg1.y = 0;
                if(patchROIimg1.height > img.size().height){
                    patchROIimg1.height = img.size().height;
                }
            }else if((patchROIimg1.y + patchROIimg1.height) > img.size().height){
                int wdiff = patchROIimg1.y + patchROIimg1.height - img.size().height;
                patchROIimg1.y -= wdiff;
                if(patchROIimg1.y < 0){
                    patchROIimg1.y = 0;
                    patchROIimg1.height = img.size().height;
                }
            }
        }
        patchfb = img(patchROIimg1);
        patchwn = patchfb.clone();
        descrDist = -1.0;
        kp2 = kp;
        kp2.pt.x -= (float)patchROIimg1.x;
        kp2.pt.y -= (float)patchROIimg1.y;

        //Change to keypoint position based on the given error range
//            if((parsMtch.descriptorType != "AKAZE") && (parsMtch.descriptorType != "KAZE")) {
        if(!forCam1) {
            distortKeyPointPosition(kp2, patchROIimg1, patchInfos.distr);
        }
//            }
        kp2err.x = kp2.pt.x + (float)patchROIimg1.x - kp.pt.x;
        kp2err.y = kp2.pt.y + (float)patchROIimg1.y - kp.pt.y;

        int itCnt = 0;
        bool noPosChange = false;
        int saltPepMinLow = 17, saltPepMaxLow = 238;
        const int saltPepMinLowMin = 5, saltPepMaxLowMax = 250;
        int saltPepMinHigh = 25, saltPepMaxHigh = 230;
        const int saltPepMinHighMax = 35, saltPepMaxHighMin = 220;
        bool try_again = false;
        do{
            Mat patchwgn;
            if((!patchInfos.useTN && parsMtch.checkDescriptorDist &&
                ((!patchInfos.takeImg2FallBack && (descrDist > patchInfos.ThTp)) ||
                 (patchInfos.takeImg2FallBack && (descrDist > 1.2 * patchInfos.ThTp))))
               || (!patchInfos.takeImg2FallBack && patchInfos.useTN && (descrDist > badDescrTH.maxVal))
               || (!patchInfos.useTN && (descrDist < 0))){
                if(!noPosChange) {
                    kp2 = kp;
                    if(!fullImgUsed) {
                        kp2.pt.x -= (float) patchROIimg1.x;
                        kp2.pt.y -= (float) patchROIimg1.y;
                    }
//                        if((parsMtch.descriptorType != "AKAZE") && (parsMtch.descriptorType != "KAZE")) {
                    if(!forCam1) {
                        distortKeyPointPosition(kp2, patchROIimg1, patchInfos.distr);
                    }
//                        }
                    kp2err.x = kp2.pt.x + (float)patchROIimg1.x - kp.pt.x;
                    kp2err.y = kp2.pt.y + (float)patchROIimg1.y - kp.pt.y;
                }

                patchwn = patchfb.clone();
                if(!patchInfos.takeImg2FallBack) {
                    meang = getRandDoubleValRng(-10.0, 10.0);
                    stdg = getRandDoubleValRng(-12.0, 12.0);
                    addImgNoiseGauss(patchwn, patchwgn, meang, stdg, visPatchNoise);
                    addImgNoiseSaltAndPepper(patchwgn, patchwn, saltPepMinLow, saltPepMaxLow, visPatchNoise);
                    saltPepMinLow--;
                    saltPepMinLow = max(saltPepMinLow, saltPepMinLowMin);
                    saltPepMaxLow++;
                    saltPepMaxLow = min(saltPepMaxLow, saltPepMaxLowMax);
                }
            }else if(!patchInfos.takeImg2FallBack){
                addImgNoiseGauss(patchwn, patchwgn, meang, stdg, visPatchNoise);
                addImgNoiseSaltAndPepper(patchwgn, patchwn, saltPepMinHigh, saltPepMaxHigh, visPatchNoise);
                saltPepMinHigh++;
                saltPepMinHigh = min(saltPepMinHigh, saltPepMinHighMax);
                saltPepMaxHigh--;
                saltPepMaxHigh = max(saltPepMaxHigh, saltPepMaxHighMin);
            }
            //Get descriptor
            vector<KeyPoint> pkp21(1, kp2);
            //pkp21[0] = kp2;
            bool kaze_noFail = true;
            if((parsMtch.descriptorType == "AKAZE") || (parsMtch.descriptorType == "KAZE")){
                kaze_noFail = getKazeProperties(patchwn, pkp21, kp2);
            }
            int err = 0;
            if(kaze_noFail) {
                err = matchinglib::getDescriptors(patchwn,
                                                  pkp21,
                                                  parsMtch.descriptorType,
                                                  descr21,
                                                  parsMtch.keyPointType);
            }else{
                err = -1;
            }
            bool itFI = false;
            if ((err != 0) || (itCnt == 15) || (itCnt == 20)) {
                if(err == 0){
                    itFI = true;
                }
                if(fullImgUsed && ((err != 0) || (itCnt == 20))){
                    //Try using the original keypoint position without location change
                    kp2 = kp;
                    kp2err = Point2f(0,0);
                    pkp21 = vector<KeyPoint>(1, kp2);

//                    Mat imgcol;
//                    cvtColor(patchwn, imgcol, cv::COLOR_GRAY2BGR);
//                    Point c((int)round(kp2.pt.x), (int)round(kp2.pt.y));
//                    cv::circle(imgcol, c, (int)round(kp2.size / 2.f), Scalar(0,0,255));
//                    namedWindow("Full image", WINDOW_AUTOSIZE);
//                    imshow("Full image", imgcol);
//
//                    waitKey(0);
//                    destroyWindow("Full image");

                    kaze_noFail = true;
                    if((parsMtch.descriptorType == "AKAZE") || (parsMtch.descriptorType == "KAZE")){
                        kaze_noFail = getKazeProperties(patchwn, pkp21, kp2);
                    }
                    if(kaze_noFail) {
                        err = matchinglib::getDescriptors(patchwn,
                                                          pkp21,
                                                          parsMtch.descriptorType,
                                                          descr21,
                                                          parsMtch.keyPointType);
                    }else{
                        err = -1;
                    }
                    if (err != 0) {
                        //Use the original descriptor
                        if(patchInfos.takeImg2FallBack){
                            descr21 = corrToIdxMap.at(corrID).getDescriptor2().clone();
                        }else {
                            cerr << "Unable to calculate a matching descriptor! Using the original one - "
                                    "this will result in a descriptor distance of 0 for this particular correspondence!"
                                 << endl;
                            descr21 = corrToIdxMap.at(corrID).getDescriptor1().clone();
                        }
                        break;
                    }else{
                        noPosChange = true;
                        itFI = false;
                    }
                }else {
                    //Use the full image instead of a patch
                    patchfb = img;
                    patchwn = patchfb.clone();
                    patchROIimg1 = Rect(Point(0,0), patchfb.size());
                    descrDist = -1.0;
                    kp2 = kp;
//                        if((parsMtch.descriptorType != "AKAZE") && (parsMtch.descriptorType != "KAZE")) {
                    if(!forCam1) {
                        distortKeyPointPosition(kp2, patchROIimg1, patchInfos.distr);
                    }
//                        }
                    kp2err.x = kp2.pt.x - kp.pt.x;
                    kp2err.y = kp2.pt.y - kp.pt.y;
                    fullImgUsed = true;
                }
            }
            if((err == 0) && !itFI){
                //Check matchability
                if(!forCam1 && !descr_cam1.empty()){
                    Mat descr1 = descr_cam1.getMat();
                    if(!descr1.empty()){
                        descrDist = getDescriptorDistance(descr1, descr21);
                    }
                    else{
                        descrDist = getDescriptorDistance(corrToIdxMap.at(corrID).getDescriptor1(), descr21);
                    }
                }else {
                    descrDist = getDescriptorDistance(corrToIdxMap.at(corrID).getDescriptor1(), descr21);
                }
            }
            itCnt++;
            bool tp_again = !patchInfos.useTN &&
                            ((!patchInfos.takeImg2FallBack && (descrDist < patchInfos.minDescrDistTP)) ||
                             (parsMtch.checkDescriptorDist && ((!patchInfos.takeImg2FallBack && (descrDist > patchInfos.ThTp)) ||
                                                               (patchInfos.takeImg2FallBack && (descrDist > 1.2 * patchInfos.ThTp)))) ||
                             (descrDist < 0));
            bool tn_again = !patchInfos.takeImg2FallBack && patchInfos.useTN &&
                            ((((combDistTNtoReal[patchInfos.i] >= 10.0) && (descrDist < patchInfos.ThTn)) ||
                              (descrDist < patchInfos.ThTnNear))
                             || (descrDist > badDescrTH.maxVal));
            try_again = tp_again || tn_again;
        }while(try_again && (itCnt < 25));
        if(itCnt >= 25){
            bool tp_again = !patchInfos.useTN &&
                            ((!patchInfos.takeImg2FallBack && (descrDist < 0.75 * patchInfos.minDescrDistTP)) ||
                             (parsMtch.checkDescriptorDist && ((!patchInfos.takeImg2FallBack && (descrDist > 1.25 * patchInfos.ThTp)) ||
                                                               (patchInfos.takeImg2FallBack && (descrDist > 1.5 * patchInfos.ThTp)))) ||
                             (descrDist < 0));
            bool tn_again = !patchInfos.takeImg2FallBack && patchInfos.useTN &&
                            ((((combDistTNtoReal[patchInfos.i] >= 10.0) && (descrDist < 0.75 * patchInfos.ThTn)) ||
                              (descrDist < 0.75 * patchInfos.ThTnNear))
                             || (descrDist > 1.2 * badDescrTH.maxVal));
            if(tp_again || tn_again) {
                //Use the original descriptor
                if(!patchInfos.takeImg2FallBack) {
                    cerr << "Unable to calculate a matching descriptor! Using the original one - "
                            "this will result in a descriptor distance of 0 for this particular correspondence!"
                         << endl;
                }
#if 1
                //Check if the descriptor extracted again without changes on the patch is the same
                if(!patchInfos.takeImg2FallBack) {
                    vector<KeyPoint> pkp21;
                    Mat desrc_tmp;
                    if (fullImgUsed) {
                        pkp21 = vector<KeyPoint>(1, kp);
                    } else {
                        KeyPoint kp2_tmp = kp;
                        kp2_tmp.pt.x -= (float) patchROIimg1.x;
                        kp2_tmp.pt.y -= (float) patchROIimg1.y;
                        pkp21 = vector<KeyPoint>(1, kp2_tmp);
                    }
                    bool kaze_noFail = true;
                    if ((parsMtch.descriptorType == "AKAZE") || (parsMtch.descriptorType == "KAZE")) {
                        KeyPoint kz_tmp = pkp21[0];
                        kaze_noFail = getKazeProperties(patchfb, pkp21, kz_tmp);
                    }
                    int err = 0;
                    if (kaze_noFail) {
                        if (matchinglib::getDescriptors(patchfb,
                                                        pkp21,
                                                        parsMtch.descriptorType,
                                                        desrc_tmp,
                                                        parsMtch.keyPointType) == 0) {
                            if (!pkp21.empty()) {
                                double descrDist_tmp;
                                descrDist_tmp = getDescriptorDistance(corrToIdxMap.at(corrID).getDescriptor1(),
                                                                      desrc_tmp);
                                if (!nearZero(descrDist_tmp)) {
                                    cerr << "SOMETHING WENT WRONG: THE USED IMAGE PATCH IS NOT THE SAME AS FOR "
                                            "CALCULATING THE INITIAL DESCRIPTOR!" << endl;
                                    if (verbose & SHOW_IMGS_AT_ERROR) {
                                        //Show correspondence in original image
                                        Mat fullimg, patchCol;
                                        cvtColor(img, fullimg, cv::COLOR_GRAY2BGR);
                                        Point c = kp_ri;
                                        cv::circle(fullimg, c, (int) round(kp.size / 2.f), Scalar(0, 0, 255));
                                        //Draw exact correspondence location
                                        cv::circle(fullimg, c, 1, Scalar(0, 255, 0));

                                        cvtColor(patchfb, patchCol, cv::COLOR_GRAY2BGR);
                                        c = Point((int) round(pkp21[0].pt.x), (int) round(pkp21[0].pt.y));
                                        cv::circle(patchCol, c, (int) round(kp.size / 2.f), Scalar(0, 0, 255));
                                        //Draw exact correspondence location
                                        cv::circle(patchCol, c, 1, Scalar(0, 255, 0));
                                        namedWindow("Original image with keypoint", WINDOW_AUTOSIZE);
                                        imshow("Original image with keypoint", fullimg);
                                        namedWindow("Patch with keypoint", WINDOW_AUTOSIZE);
                                        imshow("Patch with keypoint", patchCol);
                                        waitKey(0);
                                        destroyWindow("Original image with keypoint");
                                        destroyWindow("Patch with keypoint");
                                    }
                                }
                            }
                        }
                    }
                }
#endif
                if(forCam1 || !patchInfos.takeImg2FallBack){
                    descr21 = corrToIdxMap.at(corrID).getDescriptor1().clone();
                }else {
                    descr21 = corrToIdxMap.at(corrID).getDescriptor2().clone();
                }
                kp2 = kp;
                kp2err = Point2f(0, 0);
                descrDist = 0;
            }
        }else if(descr21.empty()){
            if(!patchInfos.takeImg2FallBack){
                cerr << "Empty descriptor found! Setting to original descriptor." << endl;
            }
            if(forCam1 || !patchInfos.takeImg2FallBack){
                descr21 = corrToIdxMap.at(corrID).getDescriptor1().clone();
            }else {
                descr21 = corrToIdxMap.at(corrID).getDescriptor2().clone();
            }
            kp2 = kp;
            kp2err = Point2f(0, 0);
            descrDist = 0;
        }
    }

    patchwn.copyTo(patchInfos.patch2);
    img(patchROIimg1).copyTo(patchInfos.patch1);
    patchInfos.patchROIimg1 = patchROIimg1;
    patchInfos.patchROIimg2 = patchROIimg2;

    return descr21.clone();
}

void showCorrespPatches(const cv::Mat &patch1, const cv::Mat &patch2){
    int border1x = 0, border1y = 0, border2x = 0, border2y = 0;
    Size patchROIimg1 = patch1.size();
    Size patchROIimg2 = patch2.size();
    if(patchROIimg1.width > patchROIimg2.width){
        border2x = patchROIimg1.width - patchROIimg2.width;
    }else{
        border1x =  patchROIimg2.width - patchROIimg1.width;
    }
    if(patchROIimg1.height > patchROIimg2.height){
        border2y = patchROIimg1.height - patchROIimg2.height;
    }else{
        border1y =  patchROIimg2.height - patchROIimg1.height;
    }
    Mat patchwc, patchc;
    if(border2x || border2y) {
        cv::copyMakeBorder(patch2, patchwc, 0, border2y, 0, border2x, BORDER_CONSTANT, Scalar::all(0));
    }else{
        patchwc = patch2;
    }
    if(border1x || border1y) {
        cv::copyMakeBorder(patch1, patchc, 0, border1y, 0, border1x, BORDER_CONSTANT, Scalar::all(0));
    }else{
        patchc = patch1;
    }
    Mat bothPathes;
    cv::hconcat(patchc, patchwc, bothPathes);
    namedWindow("Patches of first and second image", WINDOW_AUTOSIZE);
    imshow("Patches of first and second image", bothPathes);

    waitKey(0);
    destroyWindow("Patches of first and second image");
}

void genMatchSequ::distortKeyPointPosition(cv::KeyPoint &kp2,
        const cv::Rect &roi,
        std::normal_distribution<double> &distr){
    //Change to keypoint position based on the given error range
    const int usedBorder = 16;
    if(!nearZero(parsMtch.keypErrDistr.first) || !nearZero(parsMtch.keypErrDistr.second)) {
        Point2f ptm = kp2.pt;
        if (((int) round(ptm.x) < usedBorder) || ((int) round(ptm.x) > roi.width - usedBorder - 1)
            || ((int) round(ptm.y) < usedBorder) || ((int) round(ptm.y) > roi.height - usedBorder - 1)) {
            //Only change the position with enough border
            if ((((int) round(ptm.x) < usedBorder)
                 || ((int) round(ptm.x) > roi.width - usedBorder))
                && !(((int) round(ptm.y) < usedBorder)
                     || ((int) round(ptm.y) > roi.height - usedBorder))) {
                //Change only the y position and check the possible range
                double maxr = min((double) (roi.height - usedBorder - 1) - (double) ptm.y,
                                  (double) ptm.y - (double) usedBorder);
                if (maxr > (parsMtch.keypErrDistr.first + 5.0 * parsMtch.keypErrDistr.second)) {
                    float r_error = (float) abs(distr(rand_gen)) * pow(-1.f, (float) (rand2() % 2));
                    kp2.pt.y += r_error;
                } else if (maxr > 0 && !nearZero(maxr)) {
                    auto r_error = (float) getRandDoubleValRng(-1.0 * maxr, maxr);
                    kp2.pt.y += r_error;
                }
            } else if (!(((int) round(ptm.x) < usedBorder)
                         || ((int) round(ptm.x) > roi.width - usedBorder))
                       && (((int) round(ptm.y) < usedBorder)
                           || ((int) round(ptm.y) > roi.height - usedBorder))) {
                //Change only the x position and check the possible range
                double maxr = min((double) (roi.width - usedBorder - 1) - (double) ptm.x,
                                  (double) ptm.x - (double) usedBorder);
                if (maxr > (parsMtch.keypErrDistr.first + 5.0 * parsMtch.keypErrDistr.second)) {
                    float r_error = (float) abs(distr(rand_gen)) * pow(-1.f, (float) (rand2() % 2));
                    kp2.pt.x += r_error;
                } else if (maxr > 0 && !nearZero(maxr)) {
                    auto r_error = (float) getRandDoubleValRng(-1.0 * maxr, maxr);
                    kp2.pt.x += r_error;
                }
            }
        } else {
            double r_error = abs(distr(rand_gen));
            double alpha = getRandDoubleValRng(0, M_PI);
            auto xe = (float) (r_error * cos(alpha));
            auto ye = (float) (r_error * sin(alpha));
            xe += kp2.pt.x;
            ye += kp2.pt.y;
            if ((int) round(xe) < usedBorder) {
                xe = (float) usedBorder;
            } else if ((int) round(xe) > roi.width - usedBorder - 1) {
                xe = (float) (roi.width - usedBorder - 1);
            }
            if ((int) round(ye) < usedBorder) {
                ye = (float) usedBorder;
            } else if ((int) round(ye) > roi.height - usedBorder - 1) {
                ye = (float) (roi.height - usedBorder - 1);
            }
            kp2.pt.x = xe;
            kp2.pt.y = ye;
        }
    }
}

bool genMatchSequ::calcGoodBadDescriptorTH(){
    int maxComps = 0;
    vector<double> dist_bad;
    if(nrCorrsExtractWarped > 0) {
        const int typicalNr = 150;
        int compareNr = min(typicalNr, (int) keypoints1.size());
        vector<int> usedDescriptors;
        bool enlargeSelDescr = false;
        bool excludeLowerQuartile = false;

        do {
            shuffleVector(usedDescriptors, (size_t) descriptors1.rows);
            usedDescriptors.erase(usedDescriptors.begin() + compareNr, usedDescriptors.end());

            //Exclude possible true positive matches between different images
            //Get image indices
            vector<pair<size_t, int>> usedImgIdx;
            usedImgIdx.reserve((size_t) compareNr);
            for (int i = 0; i < compareNr; ++i) {
                usedImgIdx.emplace_back(make_pair(featureImgIdx[usedDescriptors[i]], i));
            }
            sort(usedImgIdx.begin(),
                 usedImgIdx.end(),
                 [](pair<size_t, int> &first, pair<size_t, int> &second) { return first.first < second.first; });
            //Group feature indices from the same images
            vector<vector<int>> multImgFeatures;
            for (int i = 0; i < compareNr - 1; ++i) {
                vector<int> sameImgs;
                int j = i + 1;
                for (; j < compareNr; ++j) {
                    sameImgs.push_back(usedImgIdx[j - 1].second);
                    if (usedImgIdx[i].first != usedImgIdx[j].first) {
                        i = j - 1;
                        break;
                    }
                }
                multImgFeatures.emplace_back(sameImgs);
                if (j == compareNr) {
                    break;
                }
            }
            if (usedImgIdx[compareNr - 2].first != usedImgIdx[compareNr - 1].first) {
                multImgFeatures.emplace_back(vector<int>(1, usedImgIdx[compareNr - 1].second));
            } else {
                multImgFeatures.back().push_back(usedImgIdx[compareNr - 1].second);
            }
            if (multImgFeatures.size() > 1) {
                //Calculate descriptor distances between different images
                vector<DMatch> descrDistMultImg;
                if (descriptors1.type() == CV_8U) {
                    for (size_t i = 0; i < multImgFeatures.size() - 1; ++i) {
                        for (size_t j = i + 1; j < multImgFeatures.size(); ++j) {
                            for (size_t k = 0; k < multImgFeatures[i].size(); ++k) {
                                const int descrIdx1 = multImgFeatures[i][k];
                                const int descrIdx11 = usedDescriptors[descrIdx1];
                                for (auto &l : multImgFeatures[j]) {
                                    double descr_norm = norm(descriptors1.row(descrIdx11),
                                                             descriptors1.row(usedDescriptors[l]),
                                                             NORM_HAMMING);
                                    descrDistMultImg.emplace_back(DMatch(descrIdx1,
                                                                         l, (float) descr_norm));
                                }
                            }
                        }
                    }
                } else {
                    for (size_t i = 0; i < multImgFeatures.size() - 1; ++i) {
                        for (size_t j = i + 1; j < multImgFeatures.size(); ++j) {
                            for (size_t k = 0; k < multImgFeatures[i].size(); ++k) {
                                const int descrIdx1 = multImgFeatures[i][k];
                                const int descrIdx11 = usedDescriptors[descrIdx1];
                                for (auto &l : multImgFeatures[j]) {
                                    double descr_norm = norm(descriptors1.row(descrIdx11),
                                                             descriptors1.row(usedDescriptors[l]),
                                                             NORM_L2);
                                    descrDistMultImg.emplace_back(DMatch(descrIdx1,
                                                                         l, (float) descr_norm));
                                }
                            }
                        }
                    }
                }

                //Calculate descriptor distances within same images
                //Check if we have features from the same images
                vector<size_t> maxNrKPSameImg;
                for (auto &i : multImgFeatures) {
                    maxNrKPSameImg.emplace_back(i.size());
                }
                size_t maxNr = *max_element(maxNrKPSameImg.begin(), maxNrKPSameImg.end());
                if (maxNr > 2) {
                    vector<float> descrDistSameImg;
                    if (descriptors1.type() == CV_8U) {
                        for (auto &i : multImgFeatures) {
                            for (size_t j = 0; j < i.size() - 1; ++j) {
                                for (size_t k = j + 1; k < i.size(); ++k) {
                                    double norm_tmp = norm(descriptors1.row(usedDescriptors[i[j]]),
                                                           descriptors1.row(usedDescriptors[i[k]]),
                                                           NORM_HAMMING);
                                    if (nearZero(norm_tmp)) {
                                        continue;
                                    }
                                    descrDistSameImg.push_back((float) norm_tmp);
                                }
                            }
                        }
                    } else {
                        for (auto &i : multImgFeatures) {
                            for (size_t j = 0; j < i.size() - 1; ++j) {
                                for (size_t k = j + 1; k < i.size(); ++k) {
                                    double norm_tmp = norm(descriptors1.row(usedDescriptors[i[j]]),
                                                           descriptors1.row(usedDescriptors[i[k]]),
                                                           NORM_L2);
                                    if (nearZero(norm_tmp)) {
                                        continue;
                                    }
                                    descrDistSameImg.push_back((float) norm_tmp);
                                }
                            }
                        }
                    }
                    //Get smallest descriptor distance within images
                    float smallestDescrDistSameImg = *min_element(descrDistSameImg.begin(), descrDistSameImg.end());

                    //Mark every descriptor distance between different images that is smaller than the smallest within an image as possible TP
                    vector<int> excludeIdx;
                    for (auto &i : descrDistMultImg) {
                        if (i.distance < smallestDescrDistSameImg) {
                            excludeIdx.push_back(i.queryIdx);
                        }
                    }

                    if (!excludeIdx.empty()) {
                        sort(excludeIdx.begin(), excludeIdx.end(),
                             [](int &first, int &second) { return first < second; });
                        //Remove duplicates
                        excludeIdx.erase(unique(excludeIdx.begin(), excludeIdx.end()), excludeIdx.end());

                        //Check the remaining number of descriptors
                        if ((((int) usedDescriptors.size() - (int) excludeIdx.size()) < 25)
                            && ((int) keypoints1.size() > max(typicalNr, compareNr))) {
                            double enlargeRat = min(1.2 * (double) usedDescriptors.size() /
                                                    (double) (usedDescriptors.size() - excludeIdx.size() + 1), 3.0);
                            int newDescrSi = (int) ceil(enlargeRat * (double) usedDescriptors.size());
                            compareNr = min(newDescrSi, (int) keypoints1.size());
                            enlargeSelDescr = true;
                            usedDescriptors.clear();
                        } else {
                            //Remove the potential true positives from the index
                            for (int i = (int) excludeIdx.size() - 1; i >= 0; i--) {
                                usedDescriptors.erase(usedDescriptors.begin() + excludeIdx[i]);
                            }
                            //Adapt the number of remaining descriptors
                            compareNr = (int) usedDescriptors.size();
                            enlargeSelDescr = false;
                        }
                    }
                } else {
                    //Exclude lower quartile
                    excludeLowerQuartile = true;
                    enlargeSelDescr = false;
                }
            }
        } while (enlargeSelDescr);

        //Calculate any possible descriptor distance of not matching descriptors
        maxComps = compareNr * (compareNr - 1) / 2;
        dist_bad = vector<double>((size_t) maxComps);
        int nrComps = 0;
        if (descriptors1.type() == CV_8U) {
            for (int i = 0; i < compareNr - 1; ++i) {
                for (int j = i + 1; j < compareNr; ++j) {
                    double descr_norm;
                    descr_norm = norm(descriptors1.row(usedDescriptors[i]), descriptors1.row(usedDescriptors[j]),
                                      NORM_HAMMING);
                    if (nearZero(descr_norm)) {
                        if (nrComps > 0) {
                            descr_norm = dist_bad[nrComps - 1];
                        } else {
                            descr_norm = 100.0;
                        }
                    }
                    dist_bad[nrComps] = descr_norm;
                    nrComps++;
                }
            }
        } else {
            for (int i = 0; i < compareNr - 1; ++i) {
                for (int j = i + 1; j < compareNr; ++j) {
                    double descr_norm;
                    descr_norm = norm(descriptors1.row(usedDescriptors[i]), descriptors1.row(usedDescriptors[j]),
                                      NORM_L2);
                    if (nearZero(descr_norm)) {
                        if (nrComps > 0) {
                            descr_norm = dist_bad[nrComps - 1];
                        } else {
                            descr_norm = 1.0;
                        }
                    }
                    dist_bad[nrComps] = descr_norm;
                    nrComps++;
                }
            }
        }

        sort(dist_bad.begin(), dist_bad.end(), [](double &first, double &second) { return first < second; });
        if (excludeLowerQuartile) {
            int newStartPos = (int) round(0.25 * (double) maxComps);
            if (maxComps - newStartPos > 25) {
                dist_bad.erase(dist_bad.begin(), dist_bad.begin() + newStartPos);
                maxComps = (int) dist_bad.size();
            }
        }

        //Get min value
        badDescrTH.minVal = *min_element(dist_bad.begin(), dist_bad.end());
    }

    if(gtmdata.isValid()){
        std::vector<double> descrDistsTN;
        bool succ = false;
        if(nrCorrsExtractWarped > 0){
            succ = getGTMDescrDistStat(descrDistsTN, badDescrTH.minVal);
        }else{
            succ = getGTMDescrDistStat(descrDistsTN);
            if(!succ){
                return false;
            }
        }
        if(succ){
            dist_bad.insert(dist_bad.end(), make_move_iterator(descrDistsTN.begin()), make_move_iterator(descrDistsTN.end()));
            badDescrTH.minVal = *min_element(dist_bad.begin(), dist_bad.end());
            maxComps = static_cast<int>(dist_bad.size());
        }
    }

    //Get max value
    badDescrTH.maxVal = *max_element(dist_bad.begin(), dist_bad.end());

    //Calculate the median
    if((maxComps % 2) == 0){
        badDescrTH.median = (dist_bad[maxComps / 2 - 1] + dist_bad[maxComps / 2]) / 2.0;
    }else{
        badDescrTH.median = dist_bad[maxComps / 2];
    }

    //Calculate the mean
    badDescrTH.mean = 0;
    for(auto& i : dist_bad){
        badDescrTH.mean += i;
    }
    badDescrTH.mean /= (double)maxComps;

    //Calculate the standard deviation
    badDescrTH.standardDev = 0;
    for(auto& i : dist_bad){
        const double diff = i - badDescrTH.mean;
        badDescrTH.standardDev += diff * diff;
    }
    badDescrTH.standardDev /= (double)maxComps;
    badDescrTH.standardDev = sqrt(badDescrTH.standardDev);
    return true;
}

bool genMatchSequ::getGTMDescrDistStat(std::vector<double> &descrDistsTN, const double &minDist){
    if(gtmdata.leftDescriptorsAll.empty() || gtmdata.rightDescriptorsAll.empty()){
        return false;
    }
    size_t nr_Elems = gtmdata.matchesGTAll.size();
    double th = 0.5 * minDist;
    gtmdata.matchesTNAll.clear();
    for (size_t i = 0; i < nr_Elems; ++i) {
        if(gtmdata.leftDescriptorsAll[i].empty() || gtmdata.rightDescriptorsAll[i].empty()){
            return false;
        }
        vector<cv::KeyPoint> kpTN1, kpTN2;
        vector<size_t> idxTN1, idxTN2;
        for (size_t k = 0; k < gtmdata.leftInlierAll[i].size(); ++k) {
            if(!gtmdata.leftInlierAll[i][k]){
                kpTN1.push_back(gtmdata.keypLAll[i][k]);
                idxTN1.push_back(k);
            }
        }
        for (size_t k = 0; k < gtmdata.rightInlierAll[i].size(); ++k) {
            if(!gtmdata.rightInlierAll[i][k]){
                kpTN2.push_back(gtmdata.keypRAll[i][k]);
                idxTN2.push_back(k);
            }
        }
        if(kpTN1.size() != kpTN2.size()){
            if(kpTN1.size() > kpTN2.size()){
                kpTN1.resize(kpTN2.size());
                idxTN1.resize(idxTN2.size());
            }else{
                kpTN2.resize(kpTN1.size());
                idxTN2.resize(idxTN1.size());
            }
        }
        vector<size_t> idx1, idx2;
        size_t tnsi = kpTN1.size();
        shuffleVector(idx1, tnsi);
        reOrderVector(kpTN1, idx1);
        shuffleVector(idx2, tnsi);
        reOrderVector(kpTN2, idx2);
        cv::Mat descriptors1N, descriptors2N;
        descriptors1N.reserve(tnsi);
        descriptors2N.reserve(tnsi);
        for(auto &j: idx1){
            descriptors1N.push_back(gtmdata.leftDescriptorsAll[i].row(idxTN1[j]));
        }
        for(auto &j: idx2){
            descriptors2N.push_back(gtmdata.rightDescriptorsAll[i].row(idxTN2[j]));
        }
        if(descriptors1N.rows != descriptors2N.rows){
            if(descriptors1N.rows > descriptors2N.rows){
                kpTN1.resize(kpTN2.size());
                descriptors1N = descriptors1N.rowRange(0, descriptors2N.rows);
                idx1.resize(kpTN2.size());
            }else{
                kpTN2.resize(kpTN1.size());
                descriptors2N = descriptors2N.rowRange(0, descriptors1N.rows);
                idx2.resize(kpTN1.size());
            }
        }
        size_t start = descrDistsTN.size();
        descrDistsTN.resize(descrDistsTN.size() + kpTN1.size());
        vector<DMatch> tnmatch;
        tnmatch.reserve(kpTN1.size());
        if(descriptors1N.type() == CV_8U) {
            for (int j = 0; j < descriptors1N.rows; ++j) {
                double descr_norm;
                descr_norm = norm(descriptors1N.row(j), descriptors2N.row(j),
                                  NORM_HAMMING);
                if(nearZero(descr_norm) || (descr_norm < th)){
                    continue;
                }
                descrDistsTN[start] = descr_norm;
                tnmatch.emplace_back(DMatch(idxTN1[idx1[j]], idxTN2[idx2[j]], static_cast<float>(descr_norm)));
                start++;
            }
        }else{
            for (int j = 0; j < descriptors1N.rows; ++j) {
                double descr_norm;
                descr_norm = norm(descriptors1N.row(j), descriptors2N.row(j), NORM_L2);
                if(nearZero(descr_norm) || (descr_norm < th)){
                    continue;
                }
                descrDistsTN[start] = descr_norm;
                tnmatch.emplace_back(DMatch(idxTN1[idx1[j]], idxTN2[idx2[j]], static_cast<float>(descr_norm)));
                start++;
            }
        }
        if(start < descrDistsTN.size()){
            descrDistsTN.erase(descrDistsTN.begin() + start, descrDistsTN.end());
        }
        gtmdata.matchesTNAll.emplace_back(std::move(tnmatch));
    }

    size_t nr_gtm = 0;
    for(auto &i: gtmdata.matchesTNAll){
        nr_gtm += i.size();
    }
    gtmdata.sum_TN = nr_gtm;
    EstimateNrTPTN(true);

    if(descrDistsTN.empty()){
        return false;
    }

    gtmdata.matchesTNAllIdx.clear();
    gtmdata.matchesTNAllIdx.reserve(nr_gtm);
    for(size_t i = 0; i < gtmdata.matchesTNAll.size(); ++i){
        for(size_t j = 0; j < gtmdata.matchesTNAll[i].size(); ++j){
            gtmdata.matchesTNAllIdx.emplace_back(i, j);
        }
    }
    vector<size_t> idx;
    shuffleVector(idx, nr_gtm);
    reOrderVector(gtmdata.matchesTNAllIdx, idx);

    return true;
}

void genMatchSequ::EstimateNrTPTN(bool reestimate){
    if(!reestimate) {
        nrTNFullSequWarped = static_cast<size_t>(round(parsMtch.WarpedPortionTN * static_cast<double>(nrTNFullSequ)));
        nrGrossTNFullSequGTM = static_cast<size_t>(round(parsMtch.portionGrossTN * static_cast<double>(nrTNFullSequ)));
        if (nrGrossTNFullSequGTM >= gtmdata.sum_TN) {
            nrGrossTNFullSequWarped = nrGrossTNFullSequGTM - gtmdata.sum_TN;
            if (nrGrossTNFullSequWarped > nrTNFullSequWarped) {
                if (verbose & PRINT_WARNING_MESSAGES) {
                    cout
                            << "Number of gross TN from warped patches is larger than specified portion of TN from warped patches "
                               "compared to GTM TN. Updating warped TN portion from " << parsMtch.WarpedPortionTN
                            << " to " <<
                            static_cast<double>(nrGrossTNFullSequWarped) / static_cast<double>(nrTNFullSequ) << endl;
                }
                nrTNFullSequWarped = nrGrossTNFullSequWarped;
            }
            nrGrossTNFullSequGTM = gtmdata.sum_TN;
        } else {
            nrGrossTNFullSequWarped = 0;
            if (nrTNFullSequWarped > nrTNFullSequ - nrGrossTNFullSequGTM) {
                nrTNFullSequWarped = nrTNFullSequ - nrGrossTNFullSequGTM;
                if (verbose & PRINT_WARNING_MESSAGES) {
                    cout << "Number of gross TN from GTM is larger than specified "
                            "portion of TN from warped patches "
                            "compared to GTM TN. Updating warped TN portion from " <<
                         parsMtch.WarpedPortionTN << " to " <<
                         static_cast<double>(nrTNFullSequWarped) /
                         static_cast<double>(nrTNFullSequ) << endl;
                }
            }
        }
        nrCorrsFullSequWarped = nrCorrsFullSequ - gtmdata.useNrTP - nrGrossTNFullSequGTM;
        nrTPFullSequWarped = nrTPFullSequ - gtmdata.useNrTP;
        //Corresponds to nrCorrsExtractWarped = 2 * nrGrossTNFullSequWarped + (nrTNFullSequWarped - nrGrossTNFullSequWarped) + nrTPFullSequWarped;
        nrCorrsExtractWarped = nrGrossTNFullSequWarped + nrTNFullSequWarped + nrTPFullSequWarped;
    }else{
        if (nrGrossTNFullSequGTM > gtmdata.sum_TN) {
            nrGrossTNFullSequGTM = gtmdata.sum_TN;
        }
    }
}

bool genMatchSequ::getRectFitsInEllipse(const cv::Mat &H,
        const cv::KeyPoint &kp,
        cv::Rect &patchROIimg1,
        cv::Rect &patchROIimg2,
        cv::Rect &patchROIimg21,
        cv::Point2d &ellipseCenter,
        double &ellipseRot,
        cv::Size2d &axes,
                                        bool &reflectionX,
                                        bool &reflectionY,
                                        cv::Size &imgFeatureSi){
    //Calculate the rotated ellipse from the keypoint size (circle) after applying the homography to the circle
    // to estimate to minimal necessary patch size
    double r = 0;
    if(kp.size <= 0) {
        r = 30.0;
    }
    else{
        r = (double) kp.size / 2.0;//Radius of the keypoint area
    }

    /*Build a matrix representation of the circle
     * see https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
     * and https://en.wikipedia.org/wiki/Conic_section
     * Generic formula for conic sections: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
     * Matrix representation: Q = [A,   B/2, D/2;
     *                             B/2, C,   E/2,
     *                             D/2, E/2, F]
     *
     * As our circle is centered arround the keypoint location, we have to take into account the shift:
     * (x - x0)^2 + (y - y0)^2 = r^2
     * => x^2 + y^2 -2x0*x -2y0*y + x0^2 + y0^2 - r^2 = 0
     */
    Mat Q = Mat::eye(3,3,CV_64FC1);
    Q.at<double>(0,2) = -1.0 * (double)kp.pt.x;
    Q.at<double>(2,0) = Q.at<double>(0,2);
    Q.at<double>(1,2) = -1.0 * (double)kp.pt.y;
    Q.at<double>(2,1) = Q.at<double>(1,2);
    Q.at<double>(2,2) = (double)(kp.pt.x * kp.pt.x + kp.pt.y * kp.pt.y) - r * r;
    //Transform the circle to get a rotated and translated ellipse
    //from https://math.stackexchange.com/questions/1572225/circle-homography
    Mat Hi = H.inv();
    Mat QH = Hi.t() * Q * Hi;
    //Check if it is an ellipse
    //see https://math.stackexchange.com/questions/280937/finding-the-angle-of-rotation-of-an-ellipse-from-its-general-equation-and-the-ot
    double chk = 4.0 * QH.at<double>(0,1) * QH.at<double>(0,1) - 4.0 * QH.at<double>(0,0) * QH.at<double>(1,1);
    if((chk > 0) || nearZero(chk)) {//We have an parabola if chk == 0 or a hyperbola if chk > 0
        return false;
    }

    //Calculate the angle of the major axis
    //see https://math.stackexchange.com/questions/280937/finding-the-angle-of-rotation-of-an-ellipse-from-its-general-equation-and-the-ot
    //and http://mathworld.wolfram.com/Ellipse.html
    double teta = 0;
    if(nearZero(QH.at<double>(0,0) - QH.at<double>(1,1)) || (nearZero(QH.at<double>(0,1)) &&
    (QH.at<double>(0,0) < QH.at<double>(1,1)))){
        teta = 0;
    }else if(nearZero(QH.at<double>(0,1)) && (QH.at<double>(0,0) > QH.at<double>(1,1))){
        teta = M_PI_2;
    }else {
        teta = std::atan(2.0 * QH.at<double>(0,1) / (QH.at<double>(0,0) - QH.at<double>(1,1))) / 2.0;
        if(QH.at<double>(0,0) > QH.at<double>(1,1)){
            teta += M_PI_2;
        }
    }
    ellipseRot = teta;

    //Calculate center of ellipse
    //First, get equation for NOT rotated ellipse
    double cosTeta = cos(teta);
    double sinTeta = sin(teta);
    double cosTeta2 = cosTeta * cosTeta;
    double sinTeta2 = sinTeta * sinTeta;
    double A1 = QH.at<double>(0,0) * cosTeta2
            + 2.0 * QH.at<double>(0,1) * cosTeta * sinTeta
            + QH.at<double>(1,1) * sinTeta2;
    //double B1 = 0;
    double C1 = QH.at<double>(0,0) * sinTeta2
            - 2.0 * QH.at<double>(0,1) * cosTeta * sinTeta
            + QH.at<double>(1,1) * cosTeta2;
    double D1 = 2.0 * QH.at<double>(0,2) * cosTeta
            + 2.0 * QH.at<double>(1,2) * sinTeta;
    double E1 = -2.0 * QH.at<double>(0,2) * sinTeta
                + 2.0 * QH.at<double>(1,2) * cosTeta;
    double F1 = QH.at<double>(2,2);

    //Get backrotated centerpoints
    double x01 = -D1 / (2.0 * A1);
    double y01 = -E1 / (2.0 * C1);
    if(!isfinite(x01) || !isfinite(y01)){
        return false;
    }
    //Get rotated (original) center points of the ellipse
    double x0 = x01 * cosTeta - y01 * sinTeta;
    double y0 = x01 * sinTeta + y01 * cosTeta;

    //Check correctness of center point
    /*Mat ckp = (Mat_<double>(3,1) << (double)kp.pt.x, (double)kp.pt.y, 1.0);
    Mat cH = H * ckp;
    cH /= cH.at<double>(2);*/

    ellipseCenter = Point2d(x0, y0);

    //Get division coefficients a^2 and b^2 of backrotated ellipse
    //(x' - x0')^2 / a^2 + (y' - y0')^2 / b^2 = 1
    double dom = -4.0 * F1 * A1 * C1 + C1 * D1 * D1 + A1 * E1 * E1;
    double a = sqrt(dom / (4.0 * A1 * C1 * C1));//Corresponds half length of major axis
    double b = sqrt(dom / (4.0 * A1 * A1 * C1));//Corresponds half length of minor axis
    if(!isfinite(a) || !isfinite(b)){
        return false;
    }
    double ab = sqrt(a * a + b * b);
    if((ab < (0.2 * r)) && ((a < 5.0) || (b < 5.0))){
        return false;
    }

    axes = Size2d(a, b);

    //Calculate 2D vector of major and minor axis
    Mat ma = (Mat_<double>(2,1) << a * cosTeta, a * sinTeta);
    Mat mi = (Mat_<double>(2,1) << -b * sinTeta, b * cosTeta);

    //Calculate corner points of rotated rectangle enclosing ellipse
    Mat corners = Mat(2,4,CV_64FC1);
    corners.col(0) = ma + mi;
    corners.col(1) = ma - mi;
    corners.col(2) = -1.0 * ma + mi;
    corners.col(3) = -1.0 * ma - mi;

    //Get dimensions of rectangle enclosing the rotated rectangle
    double minx, maxx, miny, maxy;
    cv::minMaxLoc(corners.row(0), &minx, &maxx);
    cv::minMaxLoc(corners.row(1), &miny, &maxy);
    double dimx = abs(maxx - minx);
    double dimy = abs(maxy - miny);
    if((dimx > (double)(maxPatchSizeMult2 * minPatchSize2))
    || (dimy > (double)(maxPatchSizeMult2 * minPatchSize2))){
        return false;
    }
    //Finally get the dimensions of a square into which the rectangle fits and enlarge it by 20%
    int minSquare = (int)ceil(1.2 * max(dimx, dimy));
    //Make it an odd number
    minSquare += (minSquare + 1) % 2;
    if(minSquare < minPatchSize2){
        int minImgDim = min(imgFeatureSi.width, imgFeatureSi.height);
        if(minSquare > minImgDim || minPatchSize2 > minImgDim){
            minSquare = minImgDim;
            minSquare -= (minSquare + 1) % 2;
        }else {
            minSquare = minPatchSize2;
        }
    }

    Point2i midPt((int)round(x0), (int)round(y0));
    return getImgROIs(H,
            midPt,
            minSquare,
            patchROIimg1,
            patchROIimg2,
            patchROIimg21,
            reflectionX,
            reflectionY,
            imgFeatureSi,
            kp,
            maxPatchSizeMult2);
}

bool getImgROIs(const cv::Mat &H,
                const cv::Point2i &midPt,
                const int &minSqrROIimg2,
                cv::Rect &patchROIimg1,
                cv::Rect &patchROIimg2,
                cv::Rect &patchROIimg21,
                bool &reflectionX,
                bool &reflectionY,
                const cv::Size &imgFeatureSi,
                const cv::KeyPoint &kp1,
                const int &maxPatchSizeMult2){
    int minSquare = minSqrROIimg2 + ((minSqrROIimg2 + 1) % 2);
    auto minSquare2 = (double)((minSquare - 1) / 2);
    auto minSquareMin = 0.7 * (double)minSquare;
    auto maxSquare2 = (double)(maxPatchSizeMult2 * minSquare);
    auto x0 = (double)midPt.x;
    auto y0 = (double)midPt.y;
    Mat Hi = H.inv();
    double minx, maxx, miny, maxy, dimx, dimy;

    //Transform the square back into the original image
    Mat corners1 = Mat::ones(3,4,CV_64FC1);
    x0 = round(x0);
    y0 = round(y0);
    Mat((Mat_<double>(3,1) << x0 - minSquare2, y0 - minSquare2, 1.0)).copyTo(corners1.col(0));
    Mat((Mat_<double>(3,1) << x0 + minSquare2, y0 - minSquare2, 1.0)).copyTo(corners1.col(1));
    Mat((Mat_<double>(3,1) << x0 + minSquare2, y0 + minSquare2, 1.0)).copyTo(corners1.col(2));
    Mat((Mat_<double>(3,1) << x0 - minSquare2, y0 + minSquare2, 1.0)).copyTo(corners1.col(3));
    patchROIimg2 = cv::Rect((int)floor(x0 - minSquare2 + DBL_EPSILON),
                            (int)floor(y0 - minSquare2 + DBL_EPSILON),
                            minSquare,
                            minSquare);
    Mat corners2 = Mat::ones(3,4,CV_64FC1);
    bool atBorder = false;
    for (int j = 0; j < 4; ++j) {
        corners2.col(j) = Hi * corners1.col(j);
        if (!isfinite(corners2.at<double>(0, j))
            || !isfinite(corners2.at<double>(1, j))
            || !isfinite(corners2.at<double>(2, j))) {
            return false;
        }
        corners2.col(j) /= corners2.at<double>(2,j);
        if(corners2.at<double>(0,j) < 0){
            corners2.at<double>(0,j) = 0;
            atBorder = true;
        }else if(corners2.at<double>(0,j) > (double)(imgFeatureSi.width - 1)){
            corners2.at<double>(0,j) = (double)(imgFeatureSi.width - 1);
            atBorder = true;
        }
        if(corners2.at<double>(1,j) < 0){
            corners2.at<double>(1,j) = 0;
            atBorder = true;
        }else if(corners2.at<double>(1,j) > (double)(imgFeatureSi.height - 1)){
            corners2.at<double>(1,j) = (double)(imgFeatureSi.height - 1);
            atBorder = true;
        }
    }

    //Check for reflection
    reflectionY = false;
    if(corners2.at<double>(0,0) > corners2.at<double>(0,1)){//Reflection about y-axis
        reflectionY = true;
    }
    reflectionX = false;
    if(corners2.at<double>(1,0) > corners2.at<double>(1,3)) {//Reflection about x-axis
        reflectionX = true;
    }

    //Adapt the ROI in the transformed image if we are at the border
    if(atBorder){
        for (int j = 0; j < 4; ++j) {
            corners1.col(j) = H * corners2.col(j);
            if (!isfinite(corners1.at<double>(0, j))
                || !isfinite(corners1.at<double>(1, j))
                || !isfinite(corners1.at<double>(2, j))) {
                return false;
            }
            corners1.col(j) /= corners1.at<double>(2,j);
        }
        if(corners1.at<double>(0,0) > corners1.at<double>(0,1)){//Reflection about y-axis
            minx = max(corners1.at<double>(0,1), corners1.at<double>(0,2));
            maxx = min(corners1.at<double>(0,0), corners1.at<double>(0,3));
        }
        else{
            minx = max(corners1.at<double>(0,0), corners1.at<double>(0,3));
            maxx = min(corners1.at<double>(0,1), corners1.at<double>(0,2));
        }
        if(corners1.at<double>(1,0) > corners1.at<double>(1,3)) {//Reflection about x-axis
            miny = max(corners1.at<double>(1,3), corners1.at<double>(1,2));
            maxy = min(corners1.at<double>(1,0), corners1.at<double>(1,1));
        }else{
            miny = max(corners1.at<double>(1,0), corners1.at<double>(1,1));
            maxy = min(corners1.at<double>(1,3), corners1.at<double>(1,2));
        }
        double width = maxx - minx;
        double height = maxy - miny;
        if((width < minSquareMin) || (height < minSquareMin)){
            return false;
        }
        if((width > maxSquare2) || (height > maxSquare2)){
            return false;
        }
        patchROIimg2 = cv::Rect((int)ceil(minx - DBL_EPSILON),
                                (int)ceil(miny - DBL_EPSILON),
                                (int)floor(width + DBL_EPSILON),
                                (int)floor(height + DBL_EPSILON));
    }

    //Calculate the necessary ROI in the original image
    cv::minMaxLoc(corners2.row(0), &minx, &maxx);
    cv::minMaxLoc(corners2.row(1), &miny, &maxy);
    dimx = abs(maxx - minx);
    dimy = abs(maxy - miny);
    if((dimx < 5.0) || (dimy < 5.0)){
        return false;
    }
    patchROIimg1 = cv::Rect((int)ceil(minx - DBL_EPSILON),
                            (int)ceil(miny - DBL_EPSILON),
                            (int)floor(dimx + DBL_EPSILON),
                            (int)floor(dimy + DBL_EPSILON));

    //Calculate the image ROI in the warped image holding the full patch information from the original patch
    Mat((Mat_<double>(3,1) << (double)patchROIimg1.x,
            (double)patchROIimg1.y, 1.0)).copyTo(corners2.col(0));
    Mat((Mat_<double>(3,1) << (double)(patchROIimg1.x + patchROIimg1.width - 1),
            (double)patchROIimg1.y, 1.0)).copyTo(corners2.col(1));
    Mat((Mat_<double>(3,1) << (double)(patchROIimg1.x + patchROIimg1.width - 1),
            (double)(patchROIimg1.y + patchROIimg1.height - 1), 1.0)).copyTo(corners2.col(2));
    Mat((Mat_<double>(3,1) << (double)patchROIimg1.x,
            (double)(patchROIimg1.y + patchROIimg1.height - 1), 1.0)).copyTo(corners2.col(3));
    for (int j = 0; j < 4; ++j) {
        corners1.col(j) = H * corners2.col(j);
        if (!isfinite(corners1.at<double>(0, j))
            || !isfinite(corners1.at<double>(1, j))
            || !isfinite(corners1.at<double>(2, j))) {
            return false;
        }
        corners1.col(j) /= corners1.at<double>(2,j);
    }
    cv::minMaxLoc(corners1.row(0), &minx, &maxx);
    cv::minMaxLoc(corners1.row(1), &miny, &maxy);
    dimx = abs(maxx - minx);
    dimy = abs(maxy - miny);
    if((dimx < minSquare2) || (dimy < minSquare2)){
        return false;
    }
    if((dimx > maxSquare2) || (dimy > maxSquare2)){
        return false;
    }
    patchROIimg21 = cv::Rect((int)floor(minx + DBL_EPSILON),
                             (int)floor(miny + DBL_EPSILON),
                             (int)ceil(dimx - DBL_EPSILON),
                             (int)ceil(dimy - DBL_EPSILON));
    if(patchROIimg21.x > patchROIimg2.x){
        patchROIimg21.x = patchROIimg2.x;
    }
    if(patchROIimg21.y > patchROIimg2.y){
        patchROIimg21.y = patchROIimg2.y;
    }
    if(patchROIimg21.width < (patchROIimg2.width + patchROIimg2.x - patchROIimg21.x)){
        patchROIimg21.width = patchROIimg2.width + patchROIimg2.x - patchROIimg21.x;
    }
    if(patchROIimg21.height < (patchROIimg2.height + patchROIimg2.y - patchROIimg21.y)) {
        patchROIimg21.height = patchROIimg2.height + patchROIimg2.y - patchROIimg21.y;
    }

    //Recalculate the patch ROI in the first image
    Mat((Mat_<double>(3,1) << (double)patchROIimg21.x,
            (double)patchROIimg21.y, 1.0)).copyTo(corners1.col(0));
    Mat((Mat_<double>(3,1) << (double)(patchROIimg21.x + patchROIimg21.width - 1),
            (double)patchROIimg21.y, 1.0)).copyTo(corners1.col(1));
    Mat((Mat_<double>(3,1) << (double)(patchROIimg21.x + patchROIimg21.width - 1),
            (double)(patchROIimg21.y + patchROIimg21.height - 1), 1.0)).copyTo(corners1.col(2));
    Mat((Mat_<double>(3,1) << (double)patchROIimg21.x,
            (double)(patchROIimg21.y + patchROIimg21.height - 1), 1.0)).copyTo(corners1.col(3));
    for (int j = 0; j < 4; ++j) {
        corners2.col(j) = Hi * corners1.col(j);
        if (!isfinite(corners2.at<double>(0, j))
            || !isfinite(corners2.at<double>(1, j))
            || !isfinite(corners2.at<double>(2, j))) {
            return false;
        }
        corners2.col(j) /= corners2.at<double>(2,j);
    }
    cv::Point img1LT;
    cv::Size img1WH;
    if(reflectionY && reflectionX){
        if((corners2.at<double>(0,0) < 0) || (corners2.at<double>(1,0) < 0)){
            return false;
        }
        img1LT = Point(max((int)ceil(corners2.at<double>(0,2) - DBL_EPSILON), 0),
                       max((int)ceil(corners2.at<double>(1,2) - DBL_EPSILON), 0));
    }else if(reflectionY){
        if((corners2.at<double>(0,0) < 0) || (corners2.at<double>(1,0) < 0)){
            return false;
        }
        img1LT = Point(max((int)ceil(corners2.at<double>(0,1) - DBL_EPSILON), 0),
                       max((int)ceil(corners2.at<double>(1,1) - DBL_EPSILON), 0));
    }else if(reflectionX){
        if((corners2.at<double>(0,0) < 0) || (corners2.at<double>(1,0) < 0)){
            return false;
        }
        img1LT = Point(max((int)ceil(corners2.at<double>(0,3) - DBL_EPSILON), 0),
                       max((int)ceil(corners2.at<double>(1,3) - DBL_EPSILON), 0));
    }else{
        img1LT = Point(max((int)ceil(corners2.at<double>(0,0) - DBL_EPSILON), 0),
                       max((int)ceil(corners2.at<double>(1,0) - DBL_EPSILON), 0));
    }
    img1WH = Size(patchROIimg1.x - img1LT.x + patchROIimg1.width,
                  patchROIimg1.y - img1LT.y + patchROIimg1.height);
    if((img1WH.width < 5.0)
    || (img1WH.height < 5.0)
    || ((img1LT.x + img1WH.width) > imgFeatureSi.width)
    || ((img1LT.y + img1WH.height) > imgFeatureSi.height)){
        return false;
    }
    if((img1LT.x + img1WH.width) > imgFeatureSi.width){
        img1WH.width = imgFeatureSi.width - img1LT.x;
    }
    if((img1LT.y + img1WH.height) > imgFeatureSi.height){
        img1WH.height = imgFeatureSi.height - img1LT.y;
    }

    Point kp1pt = Point((int)round(kp1.pt.x), (int)round(kp1.pt.y));
    Point distMid = kp1pt - img1LT;
    if((distMid.x < 3) || (distMid.y < 3)){
        return false;
    }
    Point newMidP = img1LT + Point(img1WH.width / 2, img1WH.height / 2);
    distMid = kp1pt - newMidP;
    if((abs(distMid.x) >= img1WH.width / 2) || (abs(distMid.y) >= img1WH.height / 2)){
        return false;
    }

    patchROIimg1 = cv::Rect(img1LT, img1WH);

    return true;
}

size_t genMatchSequ::hashFromSequPars() {
    std::stringstream ss;
    string strFromPars;

    ss << totalNrFrames;
    ss << nrCorrsFullSequ;
    ss << pars.camTrack.size();
    ss << std::setprecision(2) << pars.CorrMovObjPort;
    ss << pars.corrsPerDepth.near;
    ss << pars.corrsPerDepth.mid;
    ss << pars.corrsPerDepth.far;
    ss << pars.corrsPerRegRepRate;
    for (auto &i : pars.depthsPerRegion) {
        for (auto &j : i) {
            ss << j.near << j.mid << j.far;
        }
    }
    ss << pars.inlRatChanges;
    ss << pars.inlRatRange.first << pars.inlRatRange.second;
    ss << pars.minKeypDist;
    ss << pars.minMovObjCorrPortion;
    ss << pars.minNrMovObjs;
    ss << pars.nFramesPerCamConf;
    for (auto &i : pars.nrDepthAreasPReg) {
        for (auto &j : i) {
            ss << j.first << j.second;
        }
    }
    ss << pars.nrMovObjs;
    ss << pars.relAreaRangeMovObjs.first << pars.relAreaRangeMovObjs.second;
    ss << pars.relCamVelocity;
    ss << pars.relMovObjVelRange.first << pars.relMovObjVelRange.second;
    ss << pars.truePosChanges;
    ss << pars.truePosRange.first << pars.truePosRange.second;
    ss << pars.distortCamMat.first << pars.distortCamMat.second;

    strFromPars = ss.str();

    std::hash<std::string> hash_fn;
    return hash_fn(strFromPars);
}

size_t genMatchSequ::hashFromMtchPars() {
    std::stringstream ss;
    string strFromPars;

    ss << parsMtch.descriptorType;
    ss << parsMtch.keyPointType;
    ss << parsMtch.imgPrePostFix;
    ss << parsMtch.imgPath;
    ss << std::setprecision(2) << parsMtch.imgIntNoise.first << parsMtch.imgIntNoise.second;
    ss << parsMtch.keypErrDistr.first << parsMtch.keypErrDistr.second;
    ss << parsMtch.keypPosErrType;
//    ss << parsMtch.lostCorrPor;
    ss << parsMtch.mainStorePath;
    ss << parsMtch.storePtClouds;
    ss << parsMtch.takeLessFramesIfLessKeyP;
    ss << parsMtch.checkDescriptorDist;
    ss << std::setprecision(3) << parsMtch.repeatPatternPortStereo.first << parsMtch.repeatPatternPortStereo.second;
    ss << std::setprecision(3) << parsMtch.repeatPatternPortFToF.first << parsMtch.repeatPatternPortFToF.second;
    ss << parsMtch.distortPatchCam1;
    ss << parsMtch.oxfordGTMportion;
    ss << parsMtch.kittiGTMportion;
    ss << parsMtch.megadepthGTMportion;
    ss << parsMtch.GTMportion;
    ss << parsMtch.WarpedPortionTN;
    ss << parsMtch.portionGrossTN;
    if(!parsMtch.imageNetIDs.empty()){
        for(auto &i: parsMtch.imageNetIDs){
            ss << i;
        }
    }
    if(!parsMtch.imageNetBuzzWrds.empty()){
        for(auto &i: parsMtch.imageNetBuzzWrds){
            ss << i;
        }
    }
    ss << parsMtch.nrImgsFromImageNet;

    strFromPars = ss.str();

    std::hash<std::string> hash_fn;
    return hash_fn(strFromPars);
}

bool genMatchSequ::writeKeyPointErrorAndSrcImgs(double &meanErr, double &sdErr){
    if(!checkPathExists(matchDataPath)){
        cerr << "Given path " << matchDataPath << " to store matches does not exist!" << endl;
        return false;
    }

    string kpErrImgInfoFile = "kpErrImgInfo";
    kpErrImgInfoFile = genSequFileExtension(kpErrImgInfoFile);
    kpErrImgInfoFile = concatPath(matchDataPath, kpErrImgInfoFile);
    if(!checkOverwriteDelFiles(kpErrImgInfoFile,
                              "Output file for keypoint errors and image names already exists:", overwriteMatchingFiles)){
        return false;
    }

    FileStorage fs = FileStorage(kpErrImgInfoFile, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << kpErrImgInfoFile << endl;
        return false;
    }
    fs.writeComment("Mean and standard deviation of keypoint position errors in second stereo images");
    fs << "kpErrorsStat";
    fs << "{" << "mean" << meanErr;
    fs << "SD" << sdErr << "}";
    fs.writeComment("Image names and folders to images used to generate features and extract patches");
    vector<string> imageNames = getIDSortedImgNameList();
    fs << "imageList" << "[";
    for (auto &i : imageNames) {
        fs << i;
    }
    fs << "]";
    fs.writeComment("Statistic of the execution time for calculating the matches in microseconds");
    fs << "timeMatchStats";
    fs << "{" << "medVal" << timeMatchStats.medVal;
    fs << "arithVal" << timeMatchStats.arithVal;
    fs << "arithStd" << timeMatchStats.arithStd;
    fs << "medStd" << timeMatchStats.medStd;
    fs << "minVal" << timeMatchStats.minVal;
    fs << "maxVal" << timeMatchStats.maxVal;
    fs << "lowerQuart" << timeMatchStats.lowerQuart;
    fs << "upperQuart" << timeMatchStats.upperQuart << "}";

    fs.release();

    return true;
}

std::vector<std::string> genMatchSequ::getIDSortedImgNameList(){
    vector<std::string> imgNames;
    for(auto &i: uniqueImgIDToName){
        imgNames.push_back(*i.second);
    }
    return imgNames;
}

bool genMatchSequ::writeMatchingParameters(){
    using namespace boost::interprocess;
    if(!checkPathExists(sequParPath)){
        cerr << "Given path " << sequParPath << " to store matching parameters does not exist!" << endl;
        return false;
    }

    string matchInfoFName = "matchInfos";
    if (parsMtch.rwXMLinfo) {
        matchInfoFName += ".xml";
    } else {
        matchInfoFName += ".yaml";
    }
    string matchParsFileName = concatPath(sequParPath, matchInfoFName);
    FileStorage fs;
    int nrEntries = 0;
    string parSetNr = "parSetNr";
    try {
        named_mutex mutex(open_or_create, "write_matches_yaml");
        scoped_lock<named_mutex> lock(mutex);
        if (checkFileExists(matchParsFileName)) {
            //Check number of entries first
            if (!getNrEntriesYAML(matchParsFileName, parSetNr, nrEntries)) {
                return false;
            }
            fs = FileStorage(matchParsFileName, FileStorage::APPEND);
            if (!fs.isOpened()) {
                cerr << "Failed to open " << matchParsFileName << endl;
                return false;
            }
            fs.writeComment("\n\nNext parameters:\n");
            parSetNr += std::to_string(nrEntries);
        } else {
            fs = FileStorage(matchParsFileName, FileStorage::WRITE);
            if (!fs.isOpened()) {
                cerr << "Failed to open " << matchParsFileName << endl;
                return false;
            }
            fs.writeComment("This file contains the directory name and its corresponding parameters for "
                                "generating matches out of given 3D correspondences.\n\n");
            parSetNr += "0";
        }
        fs << parSetNr;
        fs << "{";

        fs.writeComment("Directory name (within the path containing this file) which holds matching results "
                            "using the below parameters.");
        size_t posLastSl = matchDataPath.rfind('/');
        string matchDirName = matchDataPath.substr(posLastSl + 1);
        fs << "hashMatchingPars" << matchDirName;

        fs.writeComment("Path containing the images for producing keypoint patches");
        fs << "imgPath" << parsMtch.imgPath;
        fs.writeComment("Image pre- and/or postfix for images within imgPath");
        fs << "imgPrePostFix" << parsMtch.imgPrePostFix;
        fs.writeComment("Name of keypoint detector");
        fs << "keyPointType" << parsMtch.keyPointType;
        fs.writeComment("Name of descriptor extractor");
        fs << "descriptorType" << parsMtch.descriptorType;
        fs.writeComment("Keypoint detector error (true) or error normal distribution (false)");
        fs << "keypPosErrType" << parsMtch.keypPosErrType;
        fs.writeComment("Keypoint error distribution (mean, std)");
        fs << "keypErrDistr";
        fs << "{" << "first" << parsMtch.keypErrDistr.first;
        fs << "second" << parsMtch.keypErrDistr.second << "}";
        fs.writeComment("Noise (mean, std) on the image intensity for descriptor calculation");
        fs << "imgIntNoise";
        fs << "{" << "first" << parsMtch.imgIntNoise.first;
        fs << "second" << parsMtch.imgIntNoise.second << "}";
        /*fs.writeComment("Portion (0 to 0.9) of lost correspondences from frame to frame.");
        fs << "lostCorrPor" << parsMtch.lostCorrPor;*/
        fs.writeComment("If true, all PCL point clouds and necessary information to load a cam sequence "
                            "with correspondences are stored to disk");
        fs << "storePtClouds" << parsMtch.storePtClouds;
        fs.writeComment("If true, the parameters and information are stored and read in XML format.");
        fs << "rwXMLinfo" << parsMtch.rwXMLinfo;
        fs.writeComment("If true, the stored information and parameters are compressed");
        fs << "compressedWrittenInfo" << parsMtch.compressedWrittenInfo;
        fs.writeComment("If true and too less images images are provided (resulting in too less keypoints), "
                            "only as many frames with GT matches are provided as keypoints are available.");
        fs << "takeLessFramesIfLessKeyP" << parsMtch.takeLessFramesIfLessKeyP;
        fs.writeComment("If 1, TP and TN descriptors are only accepted if their descriptor distances between "
                        "correspondences match the distribution calculated on the given images.");
        fs << "checkDescriptorDist" << parsMtch.checkDescriptorDist;
        fs.writeComment("Minimal and maximal percentage (0 to 1.0) of repeated patterns (image patches) "
                        "between stereo cameras.");
        fs << "repeatPatternPortStereo";
        fs << "{" << "first" << parsMtch.repeatPatternPortStereo.first;
        fs << "second" << parsMtch.repeatPatternPortStereo.second << "}";
        fs.writeComment("Minimal and maximal percentage (0 to 1.0) of repeated patterns (image patches) "
                        "from frame to frame.");
        fs << "repeatPatternPortFToF";
        fs << "{" << "first" << parsMtch.repeatPatternPortFToF.first;
        fs << "second" << parsMtch.repeatPatternPortFToF.second << "}";
        fs.writeComment("Enables/disables distorting a tracked image patch in the first stereo image.");
        fs << "distortPatchCam1" << parsMtch.distortPatchCam1;
        fs.writeComment("Portion of GTM from the Oxford dataset.");
        fs << "oxfordGTMportion" << parsMtch.oxfordGTMportion;
        fs.writeComment("Portion of GTM from the KITTI dataset.");
        fs << "kittiGTMportion" << parsMtch.kittiGTMportion;
        fs.writeComment("Portion of GTM from the MegaDepth dataset.");
        fs << "megadepthGTMportion" << parsMtch.megadepthGTMportion;
        fs.writeComment("Portion of GTM over all datasets (Oxford, KITTI, MegaDepth) compared to warped patch matches.");
        fs << "GTMportion" << parsMtch.GTMportion;
        fs.writeComment("Portion of TN that should be drawn from warped image patches (and not from GTM).");
        fs << "WarpedPortionTN" << parsMtch.WarpedPortionTN;
        fs.writeComment("Portion of TN that should be from GTM or from different image patches (first <-> second stereo camera).");
        fs << "portionGrossTN" << parsMtch.portionGrossTN;

        fs.release();
    }catch(interprocess_exception &ex){
        cerr << ex.what() << std::endl;
        return false;
    }

    return true;
}

bool genMatchSequ::getSequenceOverviewParsFileName(std::string &filename) const{
    if(!checkPathExists(parsMtch.mainStorePath)){
        cerr << "Given path " << parsMtch.mainStorePath <<
             " to store sequence parameter overview does not exist!" << endl;
        return false;
    }

    string matchInfoFName = "sequInfos";
    if (parsMtch.rwXMLinfo) {
        matchInfoFName += ".xml";
    } else {
        matchInfoFName += ".yaml";
    }
    filename = concatPath(parsMtch.mainStorePath, matchInfoFName);

    return true;
}

bool genMatchSequ::writeSequenceOverviewPars(){
    using namespace boost::interprocess;
    string filename;
    if(!getSequenceOverviewParsFileName(filename)){
        return false;
    }

    FileStorage fs;
    int nrEntries = 0;
    string parSetNr = "parSetNr";
    try {
        named_mutex mutex(open_or_create, "write_sequence_ov_yaml");
        scoped_lock<named_mutex> lock(mutex);
        if (checkFileExists(filename)) {
            //Check number of entries first
            if (!getNrEntriesYAML(filename, parSetNr, nrEntries)) {
                return false;
            }
            fs = FileStorage(filename, FileStorage::APPEND);
            if (!fs.isOpened()) {
                cerr << "Failed to open " << filename << endl;
                return false;
            }
            fs.writeComment("\n\nNext parameters:\n");
            parSetNr += std::to_string(nrEntries);
        } else {
            fs = FileStorage(filename, FileStorage::WRITE);
            if (!fs.isOpened()) {
                cerr << "Failed to open " << filename << endl;
                return false;
            }
            fs.writeComment("This file contains the directory name and its corresponding parameters for "
                                "generating 3D correspondences.\n\n");
            parSetNr += "0";
        }
        fs << parSetNr;
        fs << "{";

        fs.writeComment("Directory name (within the path containing this file) which holds multiple frames of "
                            "3D correspondences using the below parameters.");
        size_t posLastSl = sequParPath.rfind('/');
        string sequDirName = sequParPath.substr(posLastSl + 1);
        fs << "hashSequencePars" << sequDirName;

        writeSomeSequenceParameters(fs);
        fs << "}";

        fs.release();
    }catch(interprocess_exception &ex){
        cerr << ex.what() << std::endl;
        return false;
    }

    return true;
}

bool getNrEntriesYAML(const std::string &filename, const string &buzzword, int &nrEntries){
    FileStorage fs = FileStorage(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    nrEntries = 0;
    while(true) {
        cv::FileNode fn = fs[buzzword + std::to_string(nrEntries)];
        if (fn.empty()) {
            break;
        }
        nrEntries++;
    }

    /*for (cv::FileNodeIterator fit = fn.begin(); fit != fn.end(); ++fit)
    {
        cv::FileNode item = *fit;
        std::string somekey = item.name();
        if(somekey.find(buzzword) != string::npos){
            nrEntries++;
        }
    }*/

    fs.release();

    return true;
}

//FileStorage& operator << (FileStorage& fs, bool &value)
//{
//    if(value){
//        return (fs << 1);
//    }
//
//    return (fs << 0);
//}
//
///*void operator >> (const FileNode& n, bool& value)
//{
//    int bVal;
//    n >> bVal;
//    if(bVal){
//        value = true;
//    }else{
//        value = false;
//    }
//}*/
//
//FileStorage& operator << (FileStorage& fs, int64_t &value)
//{
//    string strVal = std::to_string(value);
//    return (fs << strVal);
//}
//
//void operator >> (const FileNode& n, int64_t& value)
//{
//    string strVal;
//    n >> strVal;
//    value = std::stoll(strVal);
//}
//
//FileNodeIterator& operator >> (FileNodeIterator& it, int64_t & value)
//{
//    *it >> value;
//    return ++it;
//}

void getRotationStats(const std::vector<cv::Mat> &Rs,
                      qualityParm &stats_roll,
                      qualityParm &stats_pitch,
                      qualityParm &stats_yaw){
    size_t nr_Rs = Rs.size();
    vector<double> roll(nr_Rs, 0), pitch(nr_Rs, 0), yaw(nr_Rs, 0);
    for(size_t i = 0; i < nr_Rs; ++i){
        getAnglesRotMat(Rs[i], roll[i], pitch[i], yaw[i], true);
    }
    getStatisticfromVec(roll, stats_roll);
    getStatisticfromVec(pitch, stats_pitch);
    getStatisticfromVec(yaw, stats_yaw);
}

void getTranslationStats(const std::vector<cv::Mat> &ts,
                         qualityParm &stats_tx,
                         qualityParm &stats_ty,
                         qualityParm &stats_tz){
    size_t nr_ts = ts.size();
    vector<double> tx(nr_ts, 0), ty(nr_ts, 0), tz(nr_ts, 0);
    for(size_t i = 0; i < nr_ts; ++i){
        tx[i] = ts[i].at<double>(0);
        ty[i] = ts[i].at<double>(1);
        tz[i] = ts[i].at<double>(2);
    }
    getStatisticfromVec(tx, stats_tx);
    getStatisticfromVec(ty, stats_ty);
    getStatisticfromVec(tz, stats_tz);
}

void genMatchSequ::getCamCoordinatesStats(double &lenght,
                            qualityParm &stats_DiffTx,
                            qualityParm &stats_DiffTy,
                            qualityParm &stats_DiffTz){
    size_t nr_ts = absCamCoordinates.size();
    vector<double> tx(nr_ts - 1, 0), ty(nr_ts - 1, 0), tz(nr_ts - 1, 0);
    lenght = 0;
    for(size_t i = 1; i < nr_ts; ++i){
        Mat tdiff = absCamCoordinates[i].t - absCamCoordinates[i-1].t;
        tx[i-1] = tdiff.at<double>(0);
        ty[i-1] = tdiff.at<double>(1);
        tz[i-1] = tdiff.at<double>(2);
        lenght += norm(tdiff);
    }
    getStatisticfromVec(tx, stats_DiffTx);
    getStatisticfromVec(ty, stats_DiffTy);
    getStatisticfromVec(tz, stats_DiffTz);
}

void genMatchSequ::writeSomeSequenceParameters(cv::FileStorage &fs){
    fs.writeComment("Number of different stereo camera configurations");
    fs << "nrStereoConfs" << (int) nrStereoConfs;
    /*fs.writeComment("Different rotations between stereo cameras");
    fs << "R" << "[";
    for (auto &i : R) {
        fs << i;
    }
    fs << "]";*/
    fs.writeComment("Statistic on rotation angles (degrees) between stereo cameras");
    qualityParm stats_roll, stats_pitch, stats_yaw;
    getRotationStats(R, stats_roll, stats_pitch, stats_yaw);
    fs << "stereo_roll_stats";
    fs << "{" << "mean" << stats_roll.arithVal;
    fs << "SD" << stats_roll.arithStd;
    fs << "min" << stats_roll.minVal;
    fs << "max" << stats_roll.maxVal << "}";
    fs << "stereo_pitch_stats";
    fs << "{" << "mean" << stats_pitch.arithVal;
    fs << "SD" << stats_pitch.arithStd;
    fs << "min" << stats_pitch.minVal;
    fs << "max" << stats_pitch.maxVal << "}";
    fs << "stereo_yaw_stats";
    fs << "{" << "mean" << stats_yaw.arithVal;
    fs << "SD" << stats_yaw.arithStd;
    fs << "min" << stats_yaw.minVal;
    fs << "max" << stats_yaw.maxVal << "}";

    /*fs.writeComment("Different translation vectors between stereo cameras");
    fs << "t" << "[";
    for (auto &i : t) {
        fs << i;
    }
    fs << "]";*/
    fs.writeComment("Statistic on translation vector elements between stereo cameras");
    qualityParm stats_tx, stats_ty, stats_tz;
    getTranslationStats(t, stats_tx, stats_ty, stats_tz);
    fs << "stereo_tx_stats";
    fs << "{" << "mean" << stats_tx.arithVal;
    fs << "SD" << stats_tx.arithStd;
    fs << "min" << stats_tx.minVal;
    fs << "max" << stats_tx.maxVal << "}";
    fs << "stereo_ty_stats";
    fs << "{" << "mean" << stats_ty.arithVal;
    fs << "SD" << stats_ty.arithStd;
    fs << "min" << stats_ty.minVal;
    fs << "max" << stats_ty.maxVal << "}";
    fs << "stereo_tz_stats";
    fs << "{" << "mean" << stats_tz.arithVal;
    fs << "SD" << stats_tz.arithStd;
    fs << "min" << stats_tz.minVal;
    fs << "max" << stats_tz.maxVal << "}";

    /*fs.writeComment("Inlier ratio for every frame");
    fs << "inlRat" << "[";
    for (auto &i : inlRat) {
        fs << i;
    }
    fs << "]";*/
    fs.writeComment("Statistic on inlier ratios");
    qualityParm stats_inlRat;
    getStatisticfromVec(inlRat, stats_inlRat);
    fs << "inlRat_stats";
    fs << "{" << "mean" << stats_inlRat.arithVal;
    fs << "SD" << stats_inlRat.arithStd;
    fs << "min" << stats_inlRat.minVal;
    fs << "max" << stats_inlRat.maxVal << "}";

    fs.writeComment("# of Frames per camera configuration");
    fs << "nFramesPerCamConf" << (int) pars.nFramesPerCamConf;
    fs.writeComment("Total number of frames in the sequence");
    fs << "totalNrFrames" << (int) totalNrFrames;
    /*fs.writeComment("Absolute number of correspondences (TP+TN) per frame");
    fs << "nrCorrs" << "[";
    for (auto &i : nrCorrs) {
        fs << (int) i;
    }
    fs << "]";*/
    qualityParm stats_nrCorrs;
    getStatisticfromVec(nrCorrs, stats_nrCorrs);
    size_t totalNrCorrs = 0;
    for (auto &i : nrCorrs){
        totalNrCorrs += i;
    }
    fs.writeComment("Statistic on the number of correspondences (TP+TN) per frame");
    fs << "nrCorrs_stats";
    fs << "{" << "mean" << stats_nrCorrs.arithVal;
    fs << "SD" << stats_nrCorrs.arithStd;
    fs << "min" << stats_nrCorrs.minVal;
    fs << "max" << stats_nrCorrs.maxVal << "}";
    fs.writeComment("Total number of correspondences (TP+TN) over all frames");
    fs << "totalNrCorrs" << (int)totalNrCorrs;

    fs.writeComment("portion of correspondences at depths");
    fs << "corrsPerDepth";
    fs << "{" << "near" << pars.corrsPerDepth.near;
    fs << "mid" << pars.corrsPerDepth.mid;
    fs << "far" << pars.corrsPerDepth.far << "}";
    /*fs.writeComment("List of portions of image correspondences at regions");
    fs << "corrsPerRegion" << "[";
    for (auto &i : pars.corrsPerRegion) {
        fs << i;
    }
    fs << "]";*/
    fs.writeComment("Mean portions of image correspondences at regions over all frames");
    Mat meanCorrsPerRegion = Mat::zeros(pars.corrsPerRegion[0].size(), pars.corrsPerRegion[0].type());
    for (auto &i : pars.corrsPerRegion) {
        meanCorrsPerRegion += i;
    }
    meanCorrsPerRegion /= (double)pars.corrsPerRegion.size();
    fs << "meanCorrsPerRegion" << meanCorrsPerRegion;

    fs.writeComment("Number of moving objects in the scene");
    fs << "nrMovObjs" << (int) pars.nrMovObjs;
    fs.writeComment("Relative area range of moving objects");
    fs << "relAreaRangeMovObjs";
    fs << "{" << "first" << pars.relAreaRangeMovObjs.first;
    fs << "second" << pars.relAreaRangeMovObjs.second << "}";
    /*fs.writeComment("Depth of moving objects.");
    fs << "movObjDepth" << "[";
    for (auto &i : pars.movObjDepth) {
        fs << (int) i;
    }
    fs << "]";*/
    fs.writeComment("Statistic on the depths of all moving objects.");
    qualityParm stats_movObjDepth;
    getStatisticfromVec(pars.movObjDepth, stats_movObjDepth);
    fs << "movObjDepth_stats";
    fs << "{" << "mean" << stats_movObjDepth.arithVal;
    fs << "SD" << stats_movObjDepth.arithStd;
    fs << "min" << stats_movObjDepth.minVal;
    fs << "max" << stats_movObjDepth.maxVal << "}";

    fs.writeComment("Minimal and maximal percentage (0 to 1.0) of random distortion of the camera matrices "
                        "K1 & K2 based on their initial values (only the focal lengths and image centers are "
                        "randomly distorted)");
    fs << "distortCamMat";
    fs << "{" << "first" << pars.distortCamMat.first;
    fs << "second" << pars.distortCamMat.second << "}";
    /*fs.writeComment(                   "Absolute coordinates of the camera centres (left or bottom cam of stereo rig) for every frame.");
    fs << "absCamCoordinates" << "[";
    for (auto &i : absCamCoordinates) {
        fs << "{" << "R" << i.R;
        fs << "t" << i.t << "}";
    }
    fs << "]";*/
    double track_lenght = 0;
    qualityParm stats_DiffTx, stats_DiffTy, stats_DiffTz;
    getCamCoordinatesStats(track_lenght, stats_DiffTx, stats_DiffTy, stats_DiffTz);
    fs.writeComment("Length of the track (moving camera centers).");
    fs << "track_lenght" << track_lenght;
    fs.writeComment("Statistic on all camera track vector elements.");
    fs << "camTrack_tx_stats";
    fs << "{" << "mean" << stats_DiffTx.arithVal;
    fs << "SD" << stats_DiffTx.arithStd;
    fs << "min" << stats_DiffTx.minVal;
    fs << "max" << stats_DiffTx.maxVal << "}";
    fs << "camTrack_ty_stats";
    fs << "{" << "mean" << stats_DiffTy.arithVal;
    fs << "SD" << stats_DiffTy.arithStd;
    fs << "min" << stats_DiffTy.minVal;
    fs << "max" << stats_DiffTy.maxVal << "}";
    fs << "camTrack_tz_stats";
    fs << "{" << "mean" << stats_DiffTz.arithVal;
    fs << "SD" << stats_DiffTz.arithStd;
    fs << "min" << stats_DiffTz.minVal;
    fs << "max" << stats_DiffTz.maxVal << "}";
}

bool genMatchSequ::writeSequenceParameters(const std::string &filename) {
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    fs.writeComment("This file contains all parameters used to generate "
                        "multiple consecutive frames with stereo correspondences.\n");

    fs.writeComment("Number of different stereo camera configurations");
    fs << "nrStereoConfs" << (int) nrStereoConfs;
    fs.writeComment("Inlier ratio for every frame");
    fs << "inlRat" << "[";
    for (auto &i : inlRat) {
        fs << i;
    }
    fs << "]";

    fs.writeComment("# of Frames per camera configuration");
    fs << "nFramesPerCamConf" << (int) pars.nFramesPerCamConf;
    fs.writeComment("Inlier ratio range");
    fs << "inlRatRange";
    fs << "{" << "first" << pars.inlRatRange.first;
    fs << "second" << pars.inlRatRange.second << "}";
    fs.writeComment("Inlier ratio change rate from pair to pair");
    fs << "inlRatChanges" << pars.inlRatChanges;
    fs.writeComment("# true positives range");
    fs << "truePosRange";
    fs << "{" << "first" << (int) pars.truePosRange.first;
    fs << "second" << (int) pars.truePosRange.second << "}";
    fs.writeComment("True positives change rate from pair to pair");
    fs << "truePosChanges" << pars.truePosChanges;
    fs.writeComment("min. distance between keypoints");
    fs << "minKeypDist" << pars.minKeypDist;
    fs.writeComment("portion of correspondences at depths");
    fs << "corrsPerDepth";
    fs << "{" << "near" << pars.corrsPerDepth.near;
    fs << "mid" << pars.corrsPerDepth.mid;
    fs << "far" << pars.corrsPerDepth.far << "}";
    fs.writeComment("List of portions of image correspondences at regions");
    fs << "corrsPerRegion" << "[";
    for (auto &i : pars.corrsPerRegion) {
        fs << i;
    }
    fs << "]";
    fs.writeComment("Repeat rate of portion of correspondences at regions.");
    fs << "corrsPerRegRepRate" << (int) pars.corrsPerRegRepRate;
    fs.writeComment("Portion of depths per region");
    fs << "depthsPerRegion" << "[";
    for (auto &i : pars.depthsPerRegion) {
        for (auto &j : i) {
            fs << "{" << "near" << j.near;
            fs << "mid" << j.mid;
            fs << "far" << j.far << "}";
        }
    }
    fs << "]";
    fs.writeComment("Min and Max number of connected depth areas per region");
    fs << "nrDepthAreasPReg" << "[";
    for (auto &i : pars.nrDepthAreasPReg) {
        for (auto &j : i) {
            fs << "{" << "first" << (int) j.first;
            fs << "second" << (int) j.second << "}";
        }
    }
    fs << "]";
    fs.writeComment("Movement direction or track of the cameras");
    fs << "camTrack" << "[";
    for (auto &i : pars.camTrack) {
        fs << i;
    }
    fs << "]";
    fs.writeComment("Relative velocity of the camera movement");
    fs << "relCamVelocity" << pars.relCamVelocity;
    fs.writeComment("Rotation matrix of the first camera centre");
    fs << "R" << pars.R;
    fs.writeComment("Number of moving objects in the scene");
    fs << "nrMovObjs" << (int) pars.nrMovObjs;
    fs.writeComment("Possible starting positions of moving objects in the image");
    fs << "startPosMovObjs" << pars.startPosMovObjs;
    fs.writeComment("Relative area range of moving objects");
    fs << "relAreaRangeMovObjs";
    fs << "{" << "first" << pars.relAreaRangeMovObjs.first;
    fs << "second" << pars.relAreaRangeMovObjs.second << "}";
    fs.writeComment("Depth of moving objects.");
    fs << "movObjDepth" << "[";
    for (auto &i : pars.movObjDepth) {
        fs << (int) i;
    }
    fs << "]";
    fs.writeComment("Movement direction of moving objects relative to camera movement");
    fs << "movObjDir" << pars.movObjDir;
    fs.writeComment("Relative velocity range of moving objects based on relative camera velocity");
    fs << "relMovObjVelRange";
    fs << "{" << "first" << pars.relMovObjVelRange.first;
    fs << "second" << pars.relMovObjVelRange.second << "}";
    fs.writeComment("Minimal portion of correspondences on moving objects for removing them");
    fs << "minMovObjCorrPortion" << pars.minMovObjCorrPortion;
    fs.writeComment("Portion of correspondences on moving objects (compared to static objects)");
    fs << "CorrMovObjPort" << pars.CorrMovObjPort;
    fs.writeComment("Minimum number of moving objects over the whole track");
    fs << "minNrMovObjs" << (int) pars.minNrMovObjs;
    fs.writeComment("Minimal and maximal percentage (0 to 1.0) of random distortion of the camera matrices "
                        "K1 & K2 based on their initial values (only the focal lengths and image centers are "
                        "randomly distorted)");
    fs << "distortCamMat";
    fs << "{" << "first" << pars.distortCamMat.first;
    fs << "second" << pars.distortCamMat.second << "}";

    fs.writeComment("Total number of frames in the sequence");
    fs << "totalNrFrames" << (int) totalNrFrames;
    fs.writeComment("User specified number of frames in the sequence");
    fs << "nTotalNrFrames" << (int) pars.nTotalNrFrames;
    fs.writeComment("Absolute number of correspondences (TP+TN) per frame");
    fs << "nrCorrs" << "[";
    for (auto &i : nrCorrs) {
        fs << (int) i;
    }
    fs << "]";
    fs.writeComment(                   "Absolute coordinates of the camera centres (left or bottom cam of stereo rig) for every frame.");
    fs << "absCamCoordinates" << "[";
    for (auto &i : absCamCoordinates) {
        fs << "{" << "R" << i.R;
        fs << "t" << i.t << "}";
    }
    fs << "]";

    fs.writeComment("Different rotations R between the 2 stereo cameras over the whole scene");
    fs << "R_stereo" << "[";
    for (auto &i : R) {
        fs << i;
    }
    fs << "]";
    fs.writeComment("Different translation vectors t between the 2 stereo cameras over the whole scene");
    fs << "t_stereo" << "[";
    for (auto &i : t) {
        fs << i;
    }
    fs << "]";

    nrMovObjAllFrames = movObj3DPtsWorldAllFrames.size();
    fs.writeComment("Number of moving object point clouds over all frames.");
    fs << "nrMovObjAllFrames" << (int) nrMovObjAllFrames;

    //Write camera parameters
    fs.writeComment("Camera matrix of cam 1");
    fs << "K1" << K1;
    fs.writeComment("Camera matrix of cam 2");
    fs << "K2" << K2;
    fs.writeComment("Image size");
    fs << "imgSize";
    fs << "{" << "width" << imgSize.width;
    fs << "height" << imgSize.height << "}";

    fs.writeComment("Statistic of the execution time for calculating the 3D sequence in microseconds");
    fs << "time3DStats";
    fs << "{" << "medVal" << time3DStats.medVal;
    fs << "arithVal" << time3DStats.arithVal;
    fs << "arithStd" << time3DStats.arithStd;
    fs << "medStd" << time3DStats.medStd;
    fs << "minVal" << time3DStats.minVal;
    fs << "maxVal" << time3DStats.maxVal;
    fs << "lowerQuart" << time3DStats.lowerQuart;
    fs << "upperQuart" << time3DStats.upperQuart << "}";

    fs.release();

    return true;
}

bool genMatchSequ::readSequenceParameters(const std::string &filename) {
    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    size_t posxml = filename.rfind(".xml");
    bool isxml = posxml != string::npos;

    int tmp = 0;
    fs["nrStereoConfs"] >> tmp;
    nrStereoConfs = (size_t)tmp;

    FileNode n = fs["inlRat"];
    FileNodeIterator it, it_end;
    inlRat.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                double inlRa1 = 0;
                n >> inlRa1;
                inlRat.push_back(inlRa1);
            } else {
                cerr << "inlRat is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                double inlRa1 = 0;
                it >> inlRa1;
                inlRat.push_back(inlRa1);
            }
        }
    }

    fs["nFramesPerCamConf"] >> tmp;
    pars3D.nFramesPerCamConf = (size_t) tmp;

    n = fs["inlRatRange"];
    double first_dbl = 0, second_dbl = 0;
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    pars3D.inlRatRange = make_pair(first_dbl, second_dbl);

    fs["inlRatChanges"] >> pars3D.inlRatChanges;

    n = fs["truePosRange"];
    int first_int = 0, second_int = 0;
    n["first"] >> first_int;
    n["second"] >> second_int;
    pars3D.truePosRange = make_pair((size_t) first_int, (size_t) second_int);

    fs["truePosChanges"] >> pars3D.truePosChanges;

    fs["minKeypDist"] >> pars3D.minKeypDist;

    n = fs["corrsPerDepth"];
    n["near"] >> pars3D.corrsPerDepth.near;
    n["mid"] >> pars3D.corrsPerDepth.mid;
    n["far"] >> pars3D.corrsPerDepth.far;

    n = fs["corrsPerRegion"];
    pars3D.corrsPerRegion.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                Mat m;
                n >> m;
                pars3D.corrsPerRegion.push_back(m.clone());
            } else {
                cerr << "corrsPerRegion is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                Mat m;
                it >> m;
                pars3D.corrsPerRegion.push_back(m.clone());
            }
        }
    }

    fs["corrsPerRegRepRate"] >> tmp;
    pars3D.corrsPerRegRepRate = (size_t) tmp;

    n = fs["depthsPerRegion"];
    if (n.type() != FileNode::SEQ) {
        cerr << "depthsPerRegion is not a sequence! FAIL" << endl;
        return false;
    }
    pars3D.depthsPerRegion = vector<vector<depthPortion>>(3, vector<depthPortion>(3));
    it = n.begin(), it_end = n.end();
    size_t idx = 0, x = 0, y = 0;
    for (; it != it_end; ++it) {
        y = idx / 3;
        x = idx % 3;

        FileNode n1 = *it;
        n1["near"] >> pars3D.depthsPerRegion[y][x].near;
        n1["mid"] >> pars3D.depthsPerRegion[y][x].mid;
        n1["far"] >> pars3D.depthsPerRegion[y][x].far;
        idx++;
    }

    n = fs["nrDepthAreasPReg"];
    if (n.type() != FileNode::SEQ) {
        cerr << "nrDepthAreasPReg is not a sequence! FAIL" << endl;
        return false;
    }
    pars3D.nrDepthAreasPReg = vector<vector<pair<size_t, size_t>>>(3, vector<pair<size_t, size_t>>(3));
    it = n.begin(), it_end = n.end();
    idx = 0;
    for (; it != it_end; ++it) {
        y = idx / 3;
        x = idx % 3;

        FileNode n1 = *it;
        n1["first"] >> first_int;
        n1["second"] >> second_int;
        pars3D.nrDepthAreasPReg[y][x] = make_pair((size_t) first_int, (size_t) second_int);
        idx++;
    }

    n = fs["camTrack"];
    pars3D.camTrack.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                Mat m;
                n >> m;
                pars3D.camTrack.emplace_back(m.clone());
            } else {
                cerr << "camTrack is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                Mat m;
                it >> m;
                pars3D.camTrack.emplace_back(m.clone());
            }
        }
    }

    fs["relCamVelocity"] >> pars3D.relCamVelocity;

    fs["R"] >> pars3D.R;

    fs["nrMovObjs"] >> tmp;
    pars3D.nrMovObjs = (size_t) tmp;

    fs["startPosMovObjs"] >> pars3D.startPosMovObjs;

    n = fs["relAreaRangeMovObjs"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    pars3D.relAreaRangeMovObjs = make_pair(first_dbl, second_dbl);

    n = fs["movObjDepth"];
    pars3D.movObjDepth.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                n >> tmp;
                pars3D.movObjDepth.push_back((depthClass) tmp);
            } else {
                cerr << "camTrack is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                it >> tmp;
                pars3D.movObjDepth.push_back((depthClass) tmp);
            }
        }
    }

    fs["movObjDir"] >> pars3D.movObjDir;

    n = fs["relMovObjVelRange"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    pars3D.relMovObjVelRange = make_pair(first_dbl, second_dbl);

    fs["minMovObjCorrPortion"] >> pars3D.minMovObjCorrPortion;

    fs["CorrMovObjPort"] >> pars3D.CorrMovObjPort;

    fs["minNrMovObjs"] >> tmp;
    pars3D.minNrMovObjs = (size_t) tmp;

    n = fs["distortCamMat"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    pars3D.distortCamMat = make_pair(first_dbl, second_dbl);

    fs["totalNrFrames"] >> tmp;
    totalNrFrames = (size_t) tmp;

    fs["nTotalNrFrames"] >> tmp;
    pars.nTotalNrFrames = (size_t) tmp;

    n = fs["nrCorrs"];
    nrCorrs.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                n >> tmp;
                nrCorrs.push_back((size_t) tmp);
            } else {
                cerr << "nrCorrs is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                it >> tmp;
                nrCorrs.push_back((size_t) tmp);
            }
        }
    }

    n = fs["absCamCoordinates"];
    absCamCoordinates.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                Mat m1, m2;
                FileNode n1 = n["R"];
                if(!n1.empty()) {
                    n1 >> m1;
                    n["t"] >> m2;
                    absCamCoordinates.emplace_back(Poses(m1.clone(), m2.clone()));
                }
            } else {
                cerr << "absCamCoordinates is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            for (; it != it_end; ++it) {
                FileNode n1 = *it;
                Mat m1, m2;
                n1["R"] >> m1;
                n1["t"] >> m2;
                absCamCoordinates.emplace_back(Poses(m1.clone(), m2.clone()));
            }
        }
    }

    n = fs["R_stereo"];
    R.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                Mat R_stereo;
                n >> R_stereo;
                R.emplace_back(R_stereo.clone());
            } else {
                cerr << "R_stereo is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                Mat R_stereo;
                it >> R_stereo;
                R.emplace_back(R_stereo.clone());
            }
        }
    }

    n = fs["t_stereo"];
    t.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                Mat t_stereo;
                n >> t_stereo;
                t.emplace_back(t_stereo.clone());
            } else {
                cerr << "t_stereo is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                Mat t_stereo;
                it >> t_stereo;
                t.emplace_back(t_stereo.clone());
            }
        }
    }

    fs["nrMovObjAllFrames"] >> tmp;
    nrMovObjAllFrames = (size_t) tmp;

    //Read camera parameters
    fs["K1"] >> K1;
    fs["K2"] >> K2;

    n = fs["imgSize"];
    n["width"] >> first_int;
    n["height"] >> second_int;
    imgSize = cv::Size(first_int, second_int);

    fs.release();

    return true;
}

bool genMatchSequ::write3DInfoSingleFrame(const std::string &filename) {
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    fs.writeComment("This file contains all correspondences of a single frame.\n");

    fs.writeComment("Frame number");
    fs << "actFrameCnt" << (int) actFrameCnt;

    fs.writeComment("Actual rotation matrix of the stereo rig: x2 = actR * x1 + actT");
    fs << "actR" << actR;

    fs.writeComment("Actual translation vector of the stereo rig: x2 = actR * x1 + actT");
    fs << "actT" << actT;

    fs.writeComment("Actual correct camera matrix of camera 1");
    fs << "K1" << K1;

    fs.writeComment("Actual correct camera matrix of camera 2");
    fs << "K2" << K2;

    fs.writeComment("Actual distorted camera matrix of camera 1");
    fs << "actKd1" << actKd1;

    fs.writeComment("Actual distorted camera matrix of camera 2");
    fs << "actKd2" << actKd2;

    fs << "actDepthNear" << actDepthNear;
    fs << "actDepthMid" << actDepthMid;
    fs << "actDepthFar" << actDepthFar;

    fs.writeComment("Combined TP correspondences (static and moving objects) of camera 1");
    fs << "combCorrsImg1TP" << combCorrsImg1TP;
    fs.writeComment("Combined TP correspondences (static and moving objects) of camera 2");
    fs << "combCorrsImg2TP" << combCorrsImg2TP;

    fs.writeComment("Combined 3D points corresponding to matches combCorrsImg1TP and combCorrsImg2TP");
    fs << "comb3DPts" << "[";
    for (auto &i : comb3DPts) {
        fs << i;
    }
    fs << "]";

    /*fs.writeComment("Index to the corresponding world 3D point within staticWorld3DPts and movObj3DPtsWorld of "
                        "combined TP correspondences (static and moving objects) in combCorrsImg1TP and "
                        "combCorrsImg2TP. Contains only the most 32 significant bits of the int64 indices.");
    fs << "combCorrsImg12TP_IdxWorld_m32bit" << "[";
    for (auto &i : combCorrsImg12TP_IdxWorld) {
        int64_t mostsig = i >> 32;
        fs << (int32_t) mostsig;
    }
    fs << "]";
    fs.writeComment("Index to the corresponding world 3D point within staticWorld3DPts and movObj3DPtsWorld of "
                        "combined TP correspondences (static and moving objects) in combCorrsImg1TP and "
                        "combCorrsImg2TP. Contains only the least 32 significant bits of the int64 indices.");
    fs << "combCorrsImg12TP_IdxWorld_l32bit" << "[";
    for (auto &i : combCorrsImg12TP_IdxWorld) {
        int64_t leastsig = (i << 32) >> 32;
        fs << (int32_t) leastsig;
    }
    fs << "]";*/

    fs.writeComment("Index to the corresponding world 3D point within staticWorld3DPts and movObj3DPtsWorld of "
                        "combined TP correspondences (static and moving objects) in combCorrsImg1TP and "
                        "combCorrsImg2TP.");
    fs << "combCorrsImg12TP_IdxWorld" << "[";
    for (auto &i : combCorrsImg12TP_IdxWorld) {
        fs << i;
    }
    fs << "]";

    fs.writeComment("Index to the corresponding world 3D point within movObj3DPtsWorldAllFrames of TP of "
                    "moving objects in combCorrsImg1TP and combCorrsImg2TP (static correspondences are positive "
                    "like in combCorrsImg12TP_IdxWorld). Indices on moving objects are negative: The first 32bit "
                    "hold the last vector index for movObj3DPtsWorldAllFrames plus 1 and the next 24bit hold the "
                    "frame number when the moving object emerged (The sum of all moving objects before this frame "
                    "number (use nrMovingObjPerFrame to calculate it) + the moving object number within this frame "
                    "number which is also included in this index lead to the correct vector index for "
                    "movObj3DPtsWorldAllFrames). The last 8bit hold the moving object number (index + 1): "
                    "idx = -1 * ((index_coordinate << 32) | (frame_number << 8) | (nr_mov_obj + 1))");
    fs << "combCorrsImg12TP_IdxWorld2" << "[";
    for (auto &i : combCorrsImg12TP_IdxWorld2) {
        fs << i;
    }
    fs << "]";

    fs.writeComment("Holds the number of visible moving objects per frame. Their sum corresponds to the number "
                    "of elements in movObj3DPtsWorldAllFrames.");
    fs << "nrMovingObjPerFrame" << "[";
    for (auto &i : nrMovingObjPerFrame) {
        fs << static_cast<int>(i);
    }
    fs << "]";

    fs.writeComment("Holds the frame count when the moving object emerged, its initial position in vector "
                    "combCorrsImg12TPContMovObj_IdxWorld, and its corresponding movObjWorldMovement: "
                    "e.g. (actFrameCnt, pos, movement_vector)");
    fs << "movObjFrameEmerge" << "[";
    for (auto &i : movObjFrameEmerge) {
        fs << "{" << "fc" << (int)get<0>(i) << "pos" << (int)get<1>(i) << "move_vect" << get<2>(i) << "}";
    }
    fs << "]";

    /*fs.writeComment("Similar to combCorrsImg12TP_IdxWorld but the vector indices for moving objects do NOT "
                        "correspond with vector elements in movObj3DPtsWorld but with a consecutive number "
                        "pointing to moving object pointclouds that were saved after they emerged. "
                        "Contains only the most 32 significant bits of the int64 indices.");
    fs << "combCorrsImg12TPContMovObj_IdxWorld_m32bit" << "[";
    for (auto &i : combCorrsImg12TPContMovObj_IdxWorld) {
        int64_t mostsig = i >> 32;
        fs << (int32_t) mostsig;
    }
    fs << "]";
    fs.writeComment("Similar to combCorrsImg12TP_IdxWorld but the vector indices for moving objects do NOT "
                        "correspond with vector elements in movObj3DPtsWorld but with a consecutive number "
                        "pointing to moving object pointclouds that were saved after they emerged. "
                        "Contains only the least 32 significant bits of the int64 indices.");
    fs << "combCorrsImg12TPContMovObj_IdxWorld_l32bit" << "[";
    for (auto &i : combCorrsImg12TPContMovObj_IdxWorld) {
        int64_t leastsig = (i << 32) >> 32;
        fs << (int32_t) leastsig;
    }
    fs << "]";*/

    fs.writeComment("Similar to combCorrsImg12TP_IdxWorld but the vector indices for moving objects do NOT "
                        "correspond with vector elements in movObj3DPtsWorld but with a consecutive number "
                        "pointing to moving object pointclouds that were saved after they emerged.");
    fs << "combCorrsImg12TPContMovObj_IdxWorld" << "[";
    for (auto &i : combCorrsImg12TPContMovObj_IdxWorld) {
        fs << i;
    }
    fs << "]";

    fs.writeComment("Combined TN correspondences (static and moving objects) in camera 1");
    fs << "combCorrsImg1TN" << combCorrsImg1TN;
    fs.writeComment("Combined TN correspondences (static and moving objects) in camera 2");
    fs << "combCorrsImg2TN" << combCorrsImg2TN;

    fs.writeComment("Number of overall TP correspondences (static and moving objects)");
    fs << "combNrCorrsTP" << combNrCorrsTP;
    fs.writeComment("Number of overall TN correspondences (static and moving objects)");
    fs << "combNrCorrsTN" << combNrCorrsTN;

    fs.writeComment("Distance values of all (static and moving objects) TN keypoint locations in the 2nd "
                        "image to the location that would be a perfect correspondence to the TN in image 1.");
    fs << "combDistTNtoReal" << "[";
    for (auto &i : combDistTNtoReal) {
        fs << i;
    }
    fs << "]";

    fs.writeComment("Final number of new generated TP correspondences for static objects.");
    fs << "finalNrTPStatCorrs" << finalNrTPStatCorrs;

    fs.writeComment("Final number of new generated TP correspondences for moving objects.");
    fs << "finalNrTPMovCorrs" << finalNrTPMovCorrs;

    fs.writeComment("Final number of backprojected TP correspondences for static objects.");
    fs << "finalNrTPStatCorrsFromLast" << finalNrTPStatCorrsFromLast;

    fs.writeComment("Final number of backprojected TP correspondences for moving objects.");
    fs << "finalNrTPMovCorrsFromLast" << finalNrTPMovCorrsFromLast;

    fs.writeComment("Final number of TN correspondences for static objects.");
    fs << "finalNrTNStatCorrs" << finalNrTNStatCorrs;

    fs.writeComment("Final number of TN correspondences for moving objects.");
    fs << "finalNrTNMovCorrs" << finalNrTNMovCorrs;

    fs.writeComment("Order of correspondences in combined Mat combCorrsImg1TP, combCorrsImg2TP, and comb3DPts");
    fs << "combCorrsImg12TPorder";
    fs << "{" << "statTPfromLast" << (int) combCorrsImg12TPorder.statTPfromLast;
    fs << "statTPnew" << (int) combCorrsImg12TPorder.statTPnew;
    fs << "movTPfromLast" << (int) combCorrsImg12TPorder.movTPfromLast;
    fs << "movTPnew" << (int) combCorrsImg12TPorder.movTPnew << "}";

    fs.writeComment("Indicates that TN correspondences of static objects are located at the beginning of Mats "
                        "combCorrsImg1TN and combCorrsImg2TN");
    /*if (combCorrsImg12TPstatFirst)
        fs << "finalNrTNMovCorrs" << 1;
    else
        fs << "finalNrTNMovCorrs" << 0;*/
    fs << "combCorrsImg12TNstatFirst" << combCorrsImg12TNstatFirst;

    fs.release();

    return true;
}

bool genMatchSequ::read3DInfoSingleFrame(const std::string &filename) {
    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    size_t posxml = filename.rfind(".xml");
    bool isxml = posxml != string::npos;

    int tmp = 0;

    fs["actFrameCnt"] >> tmp;
    actFrameCnt = (size_t) tmp;

    fs["actR"] >> actR;

    fs["actT"] >> actT;

    fs["actKd1"] >> actKd1;
    fs["actKd2"] >> actKd2;

    fs["actDepthNear"] >> actDepthNear;
    fs["actDepthMid"] >> actDepthMid;
    fs["actDepthFar"] >> actDepthFar;

    fs["combCorrsImg1TP"] >> combCorrsImg1TP;
    fs["combCorrsImg2TP"] >> combCorrsImg2TP;

    FileNode n = fs["comb3DPts"];
    comb3DPts.clear();
    FileNodeIterator it, it_end;
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                cv::Point3d pt;
                n >> pt;
                comb3DPts.push_back(pt);
            } else {
                cerr << "comb3DPts is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                cv::Point3d pt;
                it >> pt;
                comb3DPts.push_back(pt);
            }
        }
    }

    /*n = fs["combCorrsImg12TP_IdxWorld_m32bit"];
    if (n.type() != FileNode::SEQ) {
        cerr << "combCorrsImg12TP_IdxWorld_m32bit is not a sequence! FAIL" << endl;
        return false;
    }
    combCorrsImg12TP_IdxWorld.clear();
    it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        uint64_t mostsig;
        int32_t mostsig32;
        uint32_t mostsig32u;
        it >> mostsig32;
        mostsig32u = static_cast<uint32_t>(mostsig32);
        mostsig = static_cast<uint64_t>(mostsig32u);
        mostsig = mostsig << 32;
        combCorrsImg12TP_IdxWorld.push_back(static_cast<int64_t>(mostsig));
    }
    n = fs["combCorrsImg12TP_IdxWorld_l32bit"];
    if (n.type() != FileNode::SEQ) {
        cerr << "combCorrsImg12TP_IdxWorld_l32bit is not a sequence! FAIL" << endl;
        return false;
    }
    it = n.begin(), it_end = n.end();
    size_t idx = 0;
    for (; it != it_end; ++it) {
        uint64_t leastsig;
        int32_t leastsig32;
        uint32_t leastsig32u;
        it >> leastsig32;
        leastsig32u = static_cast<uint32_t>(leastsig32);
        leastsig = static_cast<uint64_t>(leastsig32u);
        leastsig = leastsig | static_cast<uint64_t>(combCorrsImg12TP_IdxWorld[idx]);
        combCorrsImg12TP_IdxWorld.push_back(static_cast<int64_t>(leastsig));
        idx++;
    }*/

    n = fs["combCorrsImg12TP_IdxWorld"];
    combCorrsImg12TP_IdxWorld.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                int64_t val;
                n >> val;
                combCorrsImg12TP_IdxWorld.push_back(val);
            } else {
                cerr << "combCorrsImg12TP_IdxWorld is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                int64_t val;
                it >> val;
                combCorrsImg12TP_IdxWorld.push_back(val);
            }
        }
    }

    n = fs["combCorrsImg12TP_IdxWorld2"];
    if(!n.empty() && n.size() > 0){
        combCorrsImg12TP_IdxWorld2.clear();
        if (n.type() != FileNode::SEQ) {
            if(isxml){
                int64_t val;
                n >> val;
                combCorrsImg12TP_IdxWorld2.push_back(val);
            }else {
                cerr << "combCorrsImg12TP_IdxWorld2 is not a sequence! FAIL" << endl;
                return false;
            }
        }else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                int64_t val;
                it >> val;
                combCorrsImg12TP_IdxWorld2.push_back(val);
            }
        }
    }

    n = fs["nrMovingObjPerFrame"];
    if(!n.empty() && n.size() > 0){
        nrMovingObjPerFrame.clear();
        if (n.type() != FileNode::SEQ) {
            if(isxml){
                int val;
                n >> val;
                nrMovingObjPerFrame.push_back(static_cast<unsigned int>(val));
            }else {
                cerr << "nrMovingObjPerFrame is not a sequence! FAIL" << endl;
                return false;
            }
        }else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                int val;
                it >> val;
                nrMovingObjPerFrame.push_back(static_cast<unsigned int>(val));
            }
        }
    }

    n = fs["movObjFrameEmerge"];
    if(!n.empty() && n.size() > 0){
        movObjFrameEmerge.clear();
        if (n.type() != FileNode::SEQ) {
            if(isxml){
                Mat move_vect;
                int fc, pos;
                FileNode n1 = n["fc"];
                if(!n1.empty()) {
                    n1 >> fc;
                    n["pos"] >> pos;
                    n["move_vect"] >> move_vect;
                    movObjFrameEmerge.emplace_back(
                            make_tuple(static_cast<size_t>(fc), static_cast<size_t>(pos), move_vect.clone()));
                }
            }else {
                cerr << "movObjFrameEmerge is not a sequence! FAIL" << endl;
                return false;
            }
        }else {
            it = n.begin(), it_end = n.end();
            for (; it != it_end; ++it) {
                FileNode n1 = *it;
                Mat move_vect;
                int fc, pos;
                n1["fc"] >> fc;
                n1["pos"] >> pos;
                n1["move_vect"] >> move_vect;
                movObjFrameEmerge.emplace_back(
                        make_tuple(static_cast<size_t>(fc), static_cast<size_t>(pos), move_vect.clone()));
            }
        }
    }

    /*n = fs["combCorrsImg12TPContMovObj_IdxWorld_m32bit"];
    if (n.type() != FileNode::SEQ) {
        cerr << "combCorrsImg12TPContMovObj_IdxWorld_m32bit is not a sequence! FAIL" << endl;
        return false;
    }
    combCorrsImg12TPContMovObj_IdxWorld.clear();
    it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        uint64_t mostsig;
        int32_t mostsig32;
        uint32_t mostsig32u;
        it >> mostsig32;
        mostsig32u = static_cast<uint32_t>(mostsig32);
        mostsig = static_cast<uint64_t>(mostsig32u);
        mostsig = mostsig << 32;
        combCorrsImg12TPContMovObj_IdxWorld.push_back(static_cast<int64_t>(mostsig));
    }
    n = fs["combCorrsImg12TPContMovObj_IdxWorld_l32bit"];
    if (n.type() != FileNode::SEQ) {
        cerr << "combCorrsImg12TPContMovObj_IdxWorld_l32bit is not a sequence! FAIL" << endl;
        return false;
    }
    it = n.begin(), it_end = n.end();
    idx = 0;
    for (; it != it_end; ++it) {
        uint64_t leastsig;
        int32_t leastsig32;
        uint32_t leastsig32u;
        it >> leastsig32;
        leastsig32u = static_cast<uint32_t>(leastsig32);
        leastsig = static_cast<uint64_t>(leastsig32u);
        leastsig = leastsig | static_cast<uint64_t>(combCorrsImg12TPContMovObj_IdxWorld[idx]);
        combCorrsImg12TPContMovObj_IdxWorld.push_back(static_cast<int64_t>(leastsig));
        idx++;
    }*/

    n = fs["combCorrsImg12TPContMovObj_IdxWorld"];
    combCorrsImg12TPContMovObj_IdxWorld.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                int64_t val;
                n >> val;
                combCorrsImg12TPContMovObj_IdxWorld.push_back(val);
            } else {
                cerr << "combCorrsImg12TPContMovObj_IdxWorld is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                int64_t val;
                it >> val;
                combCorrsImg12TPContMovObj_IdxWorld.push_back(val);
            }
        }
    }

    fs["combCorrsImg1TN"] >> combCorrsImg1TN;
    fs["combCorrsImg2TN"] >> combCorrsImg2TN;

    fs["combNrCorrsTP"] >> combNrCorrsTP;
    fs["combNrCorrsTN"] >> combNrCorrsTN;

    n = fs["combDistTNtoReal"];
    combDistTNtoReal.clear();
    if(!n.empty() && n.size() > 0) {
        if (n.type() != FileNode::SEQ) {
            if (isxml) {
                double dist = 0;
                n >> dist;
                combDistTNtoReal.push_back(dist);
            } else {
                cerr << "combDistTNtoReal is not a sequence! FAIL" << endl;
                return false;
            }
        } else {
            it = n.begin(), it_end = n.end();
            while (it != it_end) {
                double dist = 0;
                it >> dist;
                combDistTNtoReal.push_back(dist);
            }
        }
    }

    fs["finalNrTPStatCorrs"] >> finalNrTPStatCorrs;

    fs["finalNrTPMovCorrs"] >> finalNrTPMovCorrs;

    fs["finalNrTPStatCorrsFromLast"] >> finalNrTPStatCorrsFromLast;

    fs["finalNrTPMovCorrsFromLast"] >> finalNrTPMovCorrsFromLast;

    fs["finalNrTNStatCorrs"] >> finalNrTNStatCorrs;

    fs["finalNrTNMovCorrs"] >> finalNrTNMovCorrs;

    n = fs["combCorrsImg12TPorder"];
    n["statTPfromLast"] >> tmp;
    combCorrsImg12TPorder.statTPfromLast = (unsigned char) tmp;
    n["statTPnew"] >> tmp;
    combCorrsImg12TPorder.statTPnew = (unsigned char) tmp;
    n["movTPfromLast"] >> tmp;
    combCorrsImg12TPorder.movTPfromLast = (unsigned char) tmp;
    n["movTPnew"] >> tmp;
    combCorrsImg12TPorder.movTPnew = (unsigned char) tmp;

    fs["combCorrsImg12TNstatFirst"] >> tmp;
    combCorrsImg12TNstatFirst = (tmp != 0);

    fs.release();

    return true;
}

bool checkOverwriteFiles(const std::string &filename, const std::string &errmsg, bool &overwrite){
    string uip;
    if (checkFileExists(filename)) {
        cerr << errmsg << " " << filename << endl;
        cout << "Do you want to overwrite it and all the other files in this folder? (y/n)";
        cin >> uip;
        while ((uip != "y") && (uip != "n")) {
            cout << endl << "Try again:";
            cin >> uip;
        }
        cout << endl;
        if (uip == "y") {
            overwrite = true;
            if (!deleteFile(filename)) {
                cerr << "Unable to delete file. Exiting." << endl;
                return false;
            }
        } else {
            cout << "Exiting." << endl;
            return false;
        }
    }

    return true;
}

bool checkOverwriteDelFiles(const std::string &filename, const std::string &errmsg, bool &overwrite){
    if(!overwrite){
        if(!checkOverwriteFiles(filename, errmsg, overwrite))
            return false;
    } else {
        if (!deleteFile(filename)) {
            cerr << "Unable to delete file. Exiting." << endl;
            return false;
        }
    }
    return true;
}

bool genMatchSequ::writePointClouds(const std::string &path, const std::string &basename, bool &overwrite) {
    string filename = concatPath(path, basename);

    string staticWorld3DPtsFileName = filename + "_staticWorld3DPts.pcd";
    if(!checkOverwriteDelFiles(staticWorld3DPtsFileName,
            "Output file for static 3D PCL point cloud already exists:", overwrite)){
        return false;
    }
    pcl::io::savePCDFileBinaryCompressed(staticWorld3DPtsFileName, *staticWorld3DPts.get());

    for (size_t i = 0; i < movObj3DPtsWorldAllFrames.size(); ++i) {
        string fname = filename + "_movObj3DPts_" + std::to_string(i) + ".pcd";
        if(!checkOverwriteDelFiles(fname, "Output file for moving 3D PCL point cloud already exists:", overwrite)){
            return false;
        }
        pcl::io::savePCDFileBinaryCompressed(fname, movObj3DPtsWorldAllFrames[i]);
    }

    return true;
}

bool genMatchSequ::readPointClouds(const std::string &path, const std::string &basename) {
    CV_Assert(totalNrFrames > 0);

    string filename = concatPath(path, basename);
    string staticWorld3DPtsFileName = filename + "_staticWorld3DPts.pcd";

    staticWorld3DPts.reset(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile(staticWorld3DPtsFileName, *staticWorld3DPts) == -1) {
        cerr << "Could not read PCL point cloud " << staticWorld3DPtsFileName << endl;
        return false;
    }

    if (nrMovObjAllFrames > 0) {
        movObj3DPtsWorldAllFrames.clear();
        movObj3DPtsWorldAllFrames.resize(nrMovObjAllFrames);
        for (size_t i = 0; i < nrMovObjAllFrames; ++i) {
            string fname = filename + "_movObj3DPts_" + std::to_string(i) + ".pcd";
            if (pcl::io::loadPCDFile(fname, movObj3DPtsWorldAllFrames[i]) == -1) {
                cerr << "Could not read PCL point cloud " << fname << endl;
                return false;
            }
        }
    }

    return true;
}

void genMatchSequ::adaptPatchSize(){
    if(parsMtch.descriptorType == "FREAK"){
        minPatchSize2 = 151;
    }else if(parsMtch.descriptorType == "RIFF"){
        minPatchSize2 = 121;
    }
}

bool genMatchSequ::getKazeProperties(cv::Mat& patch,
                                     std::vector<cv::KeyPoint> &kp2_v,
                                     cv::KeyPoint &kp2){
    vector<KeyPoint> kps_kaze;
    bool kaze_noFail = true;
    if (matchinglib::getKeypoints(patch, kps_kaze, parsMtch.keyPointType, false) != 0) {
        if(kps_kaze.empty()){
            kaze_noFail = false;
        }
    }
    if(kaze_noFail) {
        KeyPoint kp_kaze;
        float kz_dist = FLT_MAX;
        for (auto &kz : kps_kaze) {
            float d_kp[2];
            d_kp[0] = kz.pt.x - kp2.pt.x;
            d_kp[1] = kz.pt.y - kp2.pt.y;
            d_kp[0] = d_kp[0] * d_kp[0] + d_kp[1] * d_kp[1];
            if(d_kp[0] < kz_dist){
                kz_dist = d_kp[0];
                kp_kaze = kz;
            }
        }
        if(kz_dist > 16.f){
            kaze_noFail = false;
        }else{
            kp_kaze.pt = kp2.pt;
            kp2 = kp_kaze;
            kp2_v = vector<KeyPoint>(1, kp2);
        }
    }
    return kaze_noFail;
}
