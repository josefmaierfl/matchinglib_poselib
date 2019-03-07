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

void genMatchSequ::writeSequenceParameters(const std::string &filename){
    FileStorage fs(filename, FileStorage::WRITE);

    cvWriteComment(*fs, "This file contains all parameters used to generate "
                        "multiple consecutive frames with stereo correspondences.\n", 0);

    cvWriteComment(*fs, "# of Frames per camera configuration", 0);
    fs << "nFramesPerCamConf" << pars.nFramesPerCamConf;
    cvWriteComment(*fs, "Inlier ratio range", 0);
    fs << "inlRatRange";
    fs << "{" << "first" << pars.inlRatRange.first;
    fs << "second" << pars.inlRatRange.second << "}";
    cvWriteComment(*fs, "Inlier ratio change rate from pair to pair", 0);
    fs << "inlRatChanges" << pars.inlRatChanges;
    cvWriteComment(*fs, "# true positives range", 0);
    fs << "truePosRange";
    fs << "{" << "first" << pars.truePosRange.first;
    fs << "second" << pars.truePosRange.second << "}";
    cvWriteComment(*fs, "True positives change rate from pair to pair", 0);
    fs << "truePosChanges" << pars.truePosChanges;
    cvWriteComment(*fs, "min. distance between keypoints", 0);
    fs << "minKeypDist" << pars.minKeypDist;
    cvWriteComment(*fs, "portion of correspondences at depths", 0);
    fs << "corrsPerDepth";
    fs << "{" << "near" << pars.corrsPerDepth.near;
    fs << "mid" << pars.corrsPerDepth.mid;
    fs << "far" << pars.corrsPerDepth.far << "}";
    cvWriteComment(*fs, "List of portions of image correspondences at regions", 0);
    fs << "corrsPerRegion" << "[";
    for(auto& i : pars.corrsPerRegion){
        fs << i;
    }
    fs << "]";
    cvWriteComment(*fs, "Repeat rate of portion of correspondences at regions.", 0);
    fs << "corrsPerRegRepRate" << pars.corrsPerRegRepRate;
    cvWriteComment(*fs, "Portion of depths per region", 0);
    fs << "depthsPerRegion" << "[";
    for(auto& i : pars.depthsPerRegion){
        for(auto& j : i) {
            fs << "{" << "near" << j.near;
            fs << "mid" << j.mid;
            fs << "far" << j.far << "}";
        }
    }
    fs << "]";
    cvWriteComment(*fs, "Min and Max number of connected depth areas per region", 0);
    fs << "nrDepthAreasPReg" << "[";
    for(auto& i : pars.nrDepthAreasPReg){
        for(auto& j : i) {
            fs << "{" << "first" << j.first;
            fs << "second" << j.second << "}";
        }
    }
    fs << "]";
    cvWriteComment(*fs, "Movement direction or track of the cameras", 0);
    fs << "camTrack" << "[";
    for(auto& i : pars.camTrack){
        fs << i;
    }
    fs << "]";
    cvWriteComment(*fs, "Relative velocity of the camera movement", 0);
    fs << "relCamVelocity" << pars.relCamVelocity;
    cvWriteComment(*fs, "Rotation matrix of the first camera centre", 0);
    fs << "R" << pars.R;
    cvWriteComment(*fs, "Number of moving objects in the scene", 0);
    fs << "nrMovObjs" << pars.nrMovObjs;
    cvWriteComment(*fs, "Possible starting positions of moving objects in the image", 0);
    fs << "startPosMovObjs" << pars.startPosMovObjs;
    cvWriteComment(*fs, "Relative area range of moving objects", 0);
    fs << "relAreaRangeMovObjs";
    fs << "{" << "first" << pars.relAreaRangeMovObjs.first;
    fs << "second" << pars.relAreaRangeMovObjs.second << "}";
    cvWriteComment(*fs, "Depth of moving objects.", 0);
    fs << "movObjDepth" << "[";
    for(auto& i : pars.movObjDepth){
        fs << i;
    }
    fs << "]";
    cvWriteComment(*fs, "Movement direction of moving objects relative to camera movement", 0);
    fs << "movObjDir" << pars.movObjDir;
    cvWriteComment(*fs, "Relative velocity range of moving objects based on relative camera velocity", 0);
    fs << "relMovObjVelRange";
    fs << "{" << "first" << pars.relMovObjVelRange.first;
    fs << "second" << pars.relMovObjVelRange.second << "}";
    cvWriteComment(*fs, "Minimal portion of correspondences on moving objects for removing them", 0);
    fs << "minMovObjCorrPortion" << pars.minMovObjCorrPortion;
    cvWriteComment(*fs, "Portion of correspondences on moving objects (compared to static objects)", 0);
    fs << "CorrMovObjPort" << pars.CorrMovObjPort;
    cvWriteComment(*fs, "Minimum number of moving objects over the whole track", 0);
    fs << "minNrMovObjs" << pars.minNrMovObjs;

    fs.release();
}

bool genMatchSequ::readSequenceParameters(const std::string &filename){
    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened())
    {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    fs["nFramesPerCamConf"] >> pars3D.nFramesPerCamConf;

    FileNode n = fs["inlRatRange"];
    double first_dbl, second_dbl;
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    pars3D.inlRatRange = make_pair(first_dbl, second_dbl);

    fs["inlRatChanges"] >> pars3D.inlRatChanges;

    n = fs["truePosRange"];
    size_t first_size_t, second_size_t;
    n["first"] >> first_size_t;
    n["second"] >> second_size_t;
    pars3D.truePosRange = make_pair(first_size_t, second_size_t);

    fs["truePosChanges"] >> pars3D.truePosChanges;

    fs["minKeypDist"] >> pars3D.minKeypDist;

    n = fs["corrsPerDepth"];
    n["near"] >> pars3D.corrsPerDepth.near;
    n["mid"] >> pars3D.corrsPerDepth.mid;
    n["far"] >> pars3D.corrsPerDepth.far;

    n = fs["corrsPerRegion"];
    if (n.type() != FileNode::SEQ)
    {
        cerr << "corrsPerRegion is not a sequence! FAIL" << endl;
        return false;
    }
    pars3D.corrsPerRegion.clear();
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it){
        Mat m;
        it >> m;
        pars3D.corrsPerRegion.push_back(m.clone());
    }



    fs["corrsPerRegRepRate"] >> pars3D.corrsPerRegRepRate;

    n = fs["depthsPerRegion"];
    if (n.type() != FileNode::SEQ)
    {
        cerr << "depthsPerRegion is not a sequence! FAIL" << endl;
        return false;
    }
    pars3D.depthsPerRegion = vector<vector<depthPortion>>(3, vector<depthPortion>(3));
    it = n.begin(), it_end = n.end();
    size_t idx = 0, x = 0, y = 0;
    for (; it != it_end; ++it){
        y = idx / 3;
        x = idx % 3;

        FileNode n1;
        it >> n1;
        n1["near"] >> pars3D.depthsPerRegion[y][x].near;
        n1["mid"] >> pars3D.depthsPerRegion[y][x].mid;
        n1["far"] >> pars3D.depthsPerRegion[y][x].far;
        idx++;
    }

    fs << "depthsPerRegion" << "[";
    for(auto& i : pars3D.depthsPerRegion){
        for(auto& j : i) {
            fs << "{" << "near" << j.near;
            fs << "mid" << j.mid;
            fs << "far" << j.far << "}";
        }
    }
    fs << "]";

    fs << "nrDepthAreasPReg" << "[";
    for(auto& i : pars3D.nrDepthAreasPReg){
        for(auto& j : i) {
            fs << "{" << "first" << j.first;
            fs << "second" << j.second << "}";
        }
    }
    fs << "]";

    fs << "camTrack" << "[";
    for(auto& i : pars3D.camTrack){
        fs << i;
    }
    fs << "]";

    fs << "relCamVelocity" << pars3D.relCamVelocity;

    fs << "R" << pars3D.R;

    fs << "nrMovObjs" << pars3D.nrMovObjs;

    fs << "startPosMovObjs" << pars3D.startPosMovObjs;

    fs << "relAreaRangeMovObjs";
    fs << "{" << "first" << pars3D.relAreaRangeMovObjs.first;
    fs << "second" << pars3D.relAreaRangeMovObjs.second << "}";

    fs << "movObjDepth" << "[";
    for(auto& i : pars3D.movObjDepth){
        fs << i;
    }
    fs << "]";

    fs << "movObjDir" << pars3D.movObjDir;

    fs << "relMovObjVelRange";
    fs << "{" << "first" << pars3D.relMovObjVelRange.first;
    fs << "second" << pars3D.relMovObjVelRange.second << "}";

    fs << "minMovObjCorrPortion" << pars3D.minMovObjCorrPortion;

    fs << "CorrMovObjPort" << pars3D.CorrMovObjPort;

    fs << "minNrMovObjs" << pars3D.minNrMovObjs;

    fs.release();
}
