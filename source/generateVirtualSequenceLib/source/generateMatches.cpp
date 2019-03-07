//
// Created by maierj on 06.03.19.
//

#include "generateMatches.h"
#include "io_data.h"
#include "imgFeatures.h"
#include <iomanip>

using namespace std;
using namespace cv;

//Load the images in the given folder with a given image pre- and/or postfix (supports wildcards)
bool genMatchSequ::getImageList(){
    int err = loadImageSequenceNew(parsMtch.imgPath,
            parsMtch.imgPrePostFix, imageList);
    if(err != 0){
        return false;
    }
    return true;
}

//Calculate number of TP and TN correspondences of the whole sequence
void genMatchSequ::totalNrCorrs(){
    nrCorrsFullSequ = 0;
    for(auto& i : nrCorrs){
        nrCorrsFullSequ += i;
    }
}

//Extracts the necessary number of keypoints from the set of images
bool genMatchSequ::getFeatures(){
    minNrFramesMatch = min(minNrFramesMatch, totalNrFrames);
    //Load image names
    if(!getImageList()){
        return false;
    }

    //Get random sequence of images
    vector<size_t> imgIdxs(imageList.size());
    std::shuffle(imgIdxs.begin(), imgIdxs.end(), std::mt19937{std::random_device{}()});

    //Check for the correct keypoint & descriptor types
    if(!matchinglib::IsKeypointTypeSupported(parsMtch.keyPointType)){
        cout << "Keypoint type " << parsMtch.keyPointType << " is not supported!" << endl;
        return false;
    }
    if(!matchinglib::IsDescriptorTypeSupported(parsMtch.descriptorType)){
        cout << "Descriptor type " << parsMtch.descriptorType << " is not supported!" << endl;
        return false;
    }

    //Load images and extract features & descriptors
    keypoints1.clear();
    descriptors1.release();
    imgs.reserve(imageList.size());
    int errCnt = 0;
    const int maxErrCnt = 10;
    size_t kpCnt = 0;
    for(size_t i = 0; i < imgIdxs.size(); ++i){
        //Load image
        imgs[i] = cv::imread(imageList[imgIdxs[i]],CV_LOAD_IMAGE_GRAYSCALE);
        std::vector<cv::KeyPoint> keypoints1Img;

        //Extract keypoints
        if(matchinglib::getKeypoints(imgs[i], keypoints1Img, parsMtch.keyPointType, true, 8000) != 0){
            errCnt++;
            if(errCnt > maxErrCnt){
                cout << "Extraction of keypoints failed for too many images!" << endl;
                return false;
            }
        }

        //Compute descriptors
        cv::Mat descriptors1Img;
        if(matchinglib::getDescriptors(imgs[i],
                          keypoints1Img,
                          parsMtch.descriptorType,
                          descriptors1Img,
                          parsMtch.keyPointType) != 0){
            errCnt++;
            if(errCnt > maxErrCnt){
                cout << "Calculation of descriptors failed for too many images!" << endl;
                return false;
            }
        }
        errCnt = 0;
        CV_Assert(keypoints1Img.size() == (size_t)descriptors1Img.rows);
        keypoints1.insert(keypoints1.end(), keypoints1Img.begin(), keypoints1Img.end());
        if(descriptors1.empty()){
            descriptors1 = descriptors1Img;
        }else{
            descriptors1.push_back(descriptors1Img);
        }
        kpCnt += keypoints1Img.size();
        if(kpCnt >= nrCorrsFullSequ){
            break;
        }
    }
    if(kpCnt < nrCorrsFullSequ){
        cout << "Too less keypoints - please provide additional images!";
        bool storeToDiskAndExit = !parsMtch.takeLessFramesIfLessKeyP;
        if(parsMtch.takeLessFramesIfLessKeyP){
            nrCorrsFullSequ = 0;
            size_t i = 0;
            for(; i < nrCorrs.size(); ++i){
                if((kpCnt - nrCorrs[i]) >= nrCorrsFullSequ) {
                    nrCorrsFullSequ += nrCorrs[i];
                }
                else{
                    break;
                }
            }
            nrFramesGenMatches = i;
            if(nrFramesGenMatches < minNrFramesMatch){
                cout << "Only able to generate matches for " << nrFramesGenMatches <<
                " frames but a minimum of " << minNrFramesMatch <<
                " would be required. Storing pointclouds and other information to disk for later use." << endl;
                storeToDiskAndExit = true;
            }else {
                cout << "Calculating matches for only " << nrFramesGenMatches <<
                " out of " << totalNrFrames << " frames.";
            }
        }
        if(storeToDiskAndExit){

        }
    }else{
        nrFramesGenMatches = totalNrFrames;
    }
}

size_t genMatchSequ::hashFromSequPars(){
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
    for(auto& i : pars.depthsPerRegion){
        for(auto& j : i){
            ss << j.near << j.mid << j.far;
        }
    }
    ss << pars.inlRatChanges;
    ss << pars.inlRatRange.first << pars.inlRatRange.second;
    ss << pars.minKeypDist;
    ss << pars.minMovObjCorrPortion;
    ss << pars.minNrMovObjs;
    ss << pars.nFramesPerCamConf;
    for(auto& i : pars.nrDepthAreasPReg){
        for(auto& j : i){
            ss << j.first << j.second;
        }
    }
    ss << pars.nrMovObjs;
    ss << pars.relAreaRangeMovObjs.first << pars.relAreaRangeMovObjs.second;
    ss << pars.relCamVelocity;
    ss << pars.relMovObjVelRange.first << pars.relMovObjVelRange.second;
    ss << pars.truePosChanges;
    ss << pars.truePosRange.first << pars.truePosRange.second;

    strFromPars = ss.str();

    std::hash<std::string> hash_fn;
    return hash_fn(strFromPars);
}

size_t genMatchSequ::hashFromMtchPars(){
    std::stringstream ss;
    string strFromPars;

    ss << parsMtch.descriptorType;
    ss << parsMtch.keyPointType;
    ss << parsMtch.imgPrePostFix;
    ss << parsMtch.imgPath;
    ss << std::setprecision(2) << parsMtch.imgIntNoise.first << parsMtch.imgIntNoise.second;
    ss << parsMtch.keypErrDistr.first << parsMtch.keypErrDistr.second;
    ss << parsMtch.keypPosErrType;
    ss << parsMtch.lostCorrPor;
    ss << parsMtch.mainStorePath;
    ss << parsMtch.storePtClouds;
    ss << parsMtch.takeLessFramesIfLessKeyP;

    strFromPars = ss.str();

    std::hash<std::string> hash_fn;
    return hash_fn(strFromPars);
}

void genMatchSequ::createParsHash(){
    hash_Sequ = hashFromSequPars();
    hash_Matches = hashFromMtchPars();
    hashResult = std::to_string(hash_Sequ);
    hashResult += std::to_string(hash_Matches);
}
