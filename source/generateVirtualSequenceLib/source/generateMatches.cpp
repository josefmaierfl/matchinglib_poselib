//
// Created by maierj on 06.03.19.
//

#include "generateMatches.h"
#include "io_data.h"
#include "imgFeatures.h"

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
    imgs.reserve(imageList.size());
    int errCnt = 0;
    const int maxErrCnt = 10;
    for(size_t i = 0; i < imgIdxs.size(); ++i){
        //Load image
        imgs[i] = cv::imread(imageList[imgIdxs[i]],CV_LOAD_IMAGE_GRAYSCALE);
        std::vector<cv::KeyPoint> keypoints1Img;

        //Extract keypoints
        if(matchinglib::getKeypoints(imgs[i], keypoints1Img, parsMtch.keyPointType, true, 20000) != 0){
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

    }
}
