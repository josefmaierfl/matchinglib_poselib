//
// Created by maierj on 06.03.19.
//

#include "generateMatches.h"
#include "io_data.h"
#include "imgFeatures.h"
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include "opencv2/imgproc/imgproc.hpp"

#include <pcl/visualization/pcl_visualizer.h>
#include "helper_funcs.h"
#include "side_funcs.h"

using namespace std;
using namespace cv;

/* --------------------- Function prototypes --------------------- */

bool genParsStorePath(const std::string &basePath, const std::string &subpath, std::string &resPath);
static inline FileStorage& operator << (FileStorage& fs, bool &value);
static inline void operator >> (const FileNode& n, int64_t& value);
static inline FileStorage& operator << (FileStorage& fs, int64_t &value);
static inline FileNodeIterator& operator >> (FileNodeIterator& it, int64_t & value);
bool checkOverwriteFiles(const std::string &filename, const std::string &errmsg, bool &overwrite);
bool checkOverwriteDelFiles(const std::string &filename, const std::string &errmsg, bool &overwrite);

/* -------------------------- Functions -------------------------- */

genMatchSequ::genMatchSequ(const std::string &sequLoadFolder,
                           GenMatchSequParameters &parsMtch_,
                           uint32_t verbose_) :
        genStereoSequ(verbose_),
        parsMtch(parsMtch_),
        sequParsLoaded(true){

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
    if(checkFileExists(filename)){
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
}

void genMatchSequ::genSequenceParsFileName() {
    const std::string sequParFileNameBase = "sequPars";

    sequParFileName = genSequFileExtension(sequParFileNameBase);
}

std::string genMatchSequ::genSequFileExtension(const std::string &basename){
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
        hash_Sequ = hashFromSequPars();
        if (!genParsStorePath(parsMtch.mainStorePath, std::to_string(hash_Sequ), sequParPath)) {
            return false;
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

    resPath = concatPath(subpath, subpath);
    int idx = 0;
    while(checkPathExists(resPath)){
        resPath = concatPath(subpath, subpath + "_" + std::to_string(idx));
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
        if(!writeSequenceOverviewPars()){
            cerr << "Unable to write file with sequence parameter overview!" << endl;
            return false;
        }
    }

    bool overwriteFiles = false;
    bool overWriteSuccess = true;
    while (actFrameCnt < nrFramesGenMatches){
        if(!sequParsLoaded) {
            //Generate 3D correspondences for a single frame
            startCalc_internal();
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
            if(read3DInfoSingleFrame(singleFrameDataFName)){
                cerr << "Unable to load 3D correspondence file for single frame: " <<
                     singleFrameDataFName << endl;
                return false;
            }
        }

        //Calculate matches for actual frame
        actTransGlobWorld = Mat::eye(4,4,CV_64FC1);
        absCamCoordinates[actFrameCnt].R.copyTo(actTransGlobWorld.rowRange(0,3).colRange(0,3));
        absCamCoordinates[actFrameCnt].t.copyTo(actTransGlobWorld.col(3).rowRange(0,3));
        actTransGlobWorldit = actTransGlobWorldit.inv().t();//Will be needed to transfer a plane from the local to the world coordinate system

        //Calculate the norm of the translation vector of the actual stereo configuration
        actNormT = norm(actT);



        if(sequParsLoaded){
            actFrameCnt++;
        }
    }

    if(!sequParsLoaded && parsMtch.storePtClouds && overWriteSuccess){
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

    return true;
}



void genMatchSequ::addImgNoiseGauss(const cv::Mat &patchIn, cv::Mat &patchOut, bool visualize){
    Mat mSrc_16SC;
    Mat mGaussian_noise = Mat(patchIn.size(),CV_16SC1);
    randn(mGaussian_noise,Scalar::all(parsMtch.imgIntNoise.first), Scalar::all(parsMtch.imgIntNoise.second));

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
    double z = round(zacc * getRandDoubleValRng(z_min, z_max, rand_gen)) / zacc;
    double s1 = z / b1.at<double>(2);
    Mat X = s1 * b1;
    Mat x2 = K2 * (actR * X + actT);
    x2 /= x2.at<double>(2);

    return getHomographyForDistortion(X, x1, x2, -1, 0, cv::noArray(), visualize);
}

//Checks, if a plane was already calculated for the given 3D coordinate and if yes, adapts the plane.
//If not, a new plane is calculated.
cv::Mat genMatchSequ::getHomographyForDistortionChkOld(const cv::Mat& X,
                                                 const cv::Mat& x1,
                                                 const cv::Mat& x2,
                                                 int64_t idx3D,
                                                 size_t keyPIdx,
                                                 bool visualize){
if((idx3D >= 0) && !planeTo3DIdx.empty()){
    if(planeTo3DIdx.find(idx3D) != planeTo3DIdx.end()){
        Mat trans = Mat::eye(4,4,CV_64FC1);
        trans.rowRange(0,3).colRange(0,3) = absCamCoordinates[actFrameCnt].R.t();
        trans.col(3).rowRange(0,3) = -1.0 * absCamCoordinates[actFrameCnt].R.t() * absCamCoordinates[actFrameCnt].t;
        trans = trans.inv().t();
        Mat plane = trans * planeTo3DIdx[idx3D].first;
        return getHomographyForDistortion(X, x1, x2, idx3D, keyPIdx, plane, visualize);
    }
}
return getHomographyForDistortion(X, x1, x2, idx3D, keyPIdx, cv::noArray(), visualize);
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
                                                 bool visualize){
    CV_Assert((X.rows == 3) && (X.cols == 1));
    CV_Assert((x1.rows == 3) && (x1.cols == 1));
    CV_Assert((x2.rows == 3) && (x2.cols == 1));

    Mat pn, bn, p1;
    double d;
    if(planeNVec.empty()) {
        //Get the ray to X from cam 1
        Mat b1 = K1i * x1;
        //Get the ray direction to X from cam 2
        Mat b2 = actR.t() * K2i * x2;

        //Calculate the normal vector of the plane by taking the mean vector of both camera rays
        b1 /= norm(b1);
        b2 /= norm(b2);
        bn = (b1 + b2) / 2.0;

        //Get the line on the plane normal to both viewing rays
        Mat bpa = b1.cross(b2);
        bpa /= norm(bpa);

        //Get the normal to bn and bpa
        Mat bpb = bpa.cross(bn);
        bpb /= norm(bpb);

        //Get the rotation angles
        const double maxRotAngleAlpha = 5.0 * (M_PI - acos(b1.dot(b2) / (norm(b1) * norm(b2)))) / 16.0;
        const double maxRotAngleBeta = 5.0 * M_PI / 16.0;
        //alpha ... Rotation angle about first vector 'bpa' in the plane which is perpendicular to the plane normal 'bn'
        double alpha = getRandDoubleValRng(-maxRotAngleAlpha, maxRotAngleAlpha, rand_gen);
        //beta ... Rotation angle about second vector 'bpb' in the plane which is perpendicular to the plane normal 'bn' and 'bpa'
        double beta = getRandDoubleValRng(-maxRotAngleBeta, maxRotAngleBeta, rand_gen);

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
        p1.push_back(d);
        /*Transform the plane into the world coordinate system
         * actTransGlobWorld T = [R,t;[0,0,1]]
         * actTransGlobWorldit = (T^-1)^t
         * A point p is on the plane q if p.q=0
         * p' = T p
         * q' = (T^-1)^t q
         * Point p' is on plane q' when:  p'.q'=0
         * Then: p'.q' = p^t T^t (T^-1)^t q = p^t q = p.q
        */
        if(idx3D >= 0) {
            planeTo3DIdx[idx3D] = make_pair(actTransGlobWorldit * p1, keyPIdx);
        }
    }else{
        p1 = planeNVec.getMat();
        CV_Assert((p1.rows == 4) && (p1.cols == 1));
        //Check if the plane is valid
        Mat Xh = X.clone();
        Xh.push_back(1.0);
        double errorVal = p1.dot(Xh);
        if(!nearZero(errorVal)){
            throw SequenceException("Used plane is not compatible with 3D coordinate!");
        }
        pn = p1.rowRange(0, 3).clone();
        d = p1.at<double>(3);
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
        p0.push_back(d0);
        vector<Mat> pts3D;
        pts3D.push_back(X);
        pts3D.push_back(X2);
        pts3D.push_back(X3);
        pts3D.push_back(X4);
        visualizePlanes(pts3D, p0, p1);
    }

    //Project the 3 3D points into the second image
    Mat x22 = K2 * (actR * X2 + actT);
    x22 /= x22.at<double>(2);
    Mat x23 = K2 * (actR * X3 + actT);
    x23 /= x23.at<double>(2);
    Mat x24 = K2 * (actR * X4 + actT);
    x24 /= x24.at<double>(2);

    //Calculate projective/perspective homography
    Mat x1all = Mat::ones(4,2, CV_64FC1);
    x1all.row(0) = x1.rowRange(0,2).t();
    x1all.row(1) = x12.rowRange(0,2).t();
    x1all.row(2) = x13.rowRange(0,2).t();
    x1all.row(3) = x14.rowRange(0,2).t();
    Mat x2all = Mat::ones(4,2, CV_64FC1);
    x2all.row(0) = x2.rowRange(0,2).t();
    x2all.row(1) = x22.rowRange(0,2).t();
    x2all.row(2) = x23.rowRange(0,2).t();
    x2all.row(3) = x24.rowRange(0,2).t();
    Mat H = getPerspectiveTransform(x1all, x2all);
    //Eliminate the translation
    Mat tm = H * x1;
    tm /= tm.at<double>(2);
    tm = x1 - tm;
    Mat tback = Mat::eye(3,3,CV_64FC1);
    tback.at<double>(0,2) = tm.at<double>(0);
    tback.at<double>(1,2) = tm.at<double>(1);
    H = tback * H;
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
    /*viewer->addPlane(plane_coeff,
                     (float)pts3D[0].at<double>(0),
                     (float)pts3D[0].at<double>(1),
                     (float)pts3D[0].at<double>(2),
                     "initPlane");*/
    viewer->addPlane(plane_coeff,
                     "initPlane");
    plane_coeff.values[0] = (float)plane2.at<double>(0);
    plane_coeff.values[1] = (float)plane2.at<double>(1);
    plane_coeff.values[2] = (float)plane2.at<double>(2);
    plane_coeff.values[3] = (float)plane2.at<double>(3);
    /*viewer->addPlane(plane_coeff,
                     (float)pts3D[0].at<double>(0),
                     (float)pts3D[0].at<double>(1),
                     (float)pts3D[0].at<double>(2),
                     "resPlane");*/
    viewer->addPlane(plane_coeff,
                     "resPlane");

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
    double &siny = a_.at<double>(0);
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
    int err = loadImageSequenceNew(parsMtch.imgPath,
                                   parsMtch.imgPrePostFix, imageList);

    return (err == 0);
}

//Calculate number of TP and TN correspondences of the whole sequence
void genMatchSequ::totalNrCorrs() {
    nrCorrsFullSequ = 0;
    for (auto &i : nrCorrs) {
        nrCorrsFullSequ += i;
    }
}

//Extracts the necessary number of keypoints from the set of images
bool genMatchSequ::getFeatures() {
    minNrFramesMatch = max(min(minNrFramesMatch, totalNrFrames / 2), static_cast<size_t>(1));
    //Load image names
    if (!getImageList()) {
        return false;
    }
    size_t nrImgs = imageList.size();

    //Get random sequence of images
    vector<size_t> imgIdxs(nrImgs);
    std::shuffle(imgIdxs.begin(), imgIdxs.end(), std::mt19937{std::random_device{}()});
    vector<string> imageList_tmp;
    imageList_tmp.reserve(nrImgs);
    for(auto& i : imageList){
        imageList_tmp.emplace_back(std::move(i));
    }
    imageList = std::move(imageList_tmp);

    //Check for the correct keypoint & descriptor types
    if (!matchinglib::IsKeypointTypeSupported(parsMtch.keyPointType)) {
        cout << "Keypoint type " << parsMtch.keyPointType << " is not supported!" << endl;
        return false;
    }
    if (!matchinglib::IsDescriptorTypeSupported(parsMtch.descriptorType)) {
        cout << "Descriptor type " << parsMtch.descriptorType << " is not supported!" << endl;
        return false;
    }

    //Load images and extract features & descriptors
    keypoints1.clear();
    descriptors1.release();
//    imgs.reserve(imageList.size());
    int errCnt = 0;
    const int maxErrCnt = 10;
    size_t kpCnt = 0;
    if(nrImgs <= maxImgLoad) {
        imgs.reserve(nrImgs);
    }
    for (size_t i = 0; i < nrImgs; ++i) {
        //Load image
        Mat img;
        if(nrImgs <= maxImgLoad){
            imgs.emplace_back(cv::imread(imageList[i], CV_LOAD_IMAGE_GRAYSCALE));
            img = imgs.back();
        }
        else{
            img = cv::imread(imageList[i], CV_LOAD_IMAGE_GRAYSCALE);
        }
//        imgs.emplace_back(cv::imread(imageList[i], CV_LOAD_IMAGE_GRAYSCALE));
        std::vector<cv::KeyPoint> keypoints1Img;

        //Extract keypoints
        if (matchinglib::getKeypoints(img, keypoints1Img, parsMtch.keyPointType, true, 8000) != 0) {
            errCnt++;
            if (errCnt > maxErrCnt) {
                cout << "Extraction of keypoints failed for too many images!" << endl;
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
        if (kpCnt >= nrCorrsFullSequ) {
            break;
        }
    }

    //Shuffle keypoints and descriptors
    vector<KeyPoint> keypoints1_tmp(keypoints1.size());
    Mat descriptors1_tmp = Mat(descriptors1.rows, descriptors1.cols, descriptors1.type());
    vector<size_t> featureImgIdx_tmp(featureImgIdx.size());
    vector<size_t> featureIdxs(keypoints1.size());
    std::shuffle(featureIdxs.begin(), featureIdxs.end(), std::mt19937{std::random_device{}()});
    for(int i = 0; i < (int)featureIdxs.size(); ++i){
        keypoints1_tmp[i] = std::move(keypoints1[featureIdxs[i]]);
        descriptors1.row((int)featureIdxs[i]).copyTo(descriptors1_tmp.row(i));
        featureImgIdx_tmp[i] = std::move(featureImgIdx[featureIdxs[i]]);
    }
    keypoints1 = std::move(keypoints1_tmp);
    descriptors1 = descriptors1_tmp;
    featureImgIdx = std::move(featureImgIdx_tmp);


    if (kpCnt < nrCorrsFullSequ) {
        cout << "Too less keypoints - please provide additional images! "
        << nrCorrsFullSequ << " features are required but only "
        << kpCnt << " could be extracted from the images." << endl;
        if (parsMtch.takeLessFramesIfLessKeyP) {
            nrCorrsFullSequ = 0;
            size_t i = 0;
            for (; i < nrCorrs.size(); ++i) {
                if ((kpCnt - nrCorrs[i]) >= nrCorrsFullSequ) {
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
                     " out of " << totalNrFrames << " frames.";
            }
        }
    } else {
        nrFramesGenMatches = totalNrFrames;
    }

    //Get most used keypoint images per frame to avoid holding all images in memory in the case a huge amount of images is used
    //Thus, only the maxImgLoad images from which the most keypoints for a single frame are used are loaded into memory
    if(nrImgs > maxImgLoad){
        size_t featureIdx = 0;
        loadImgsEveryFrame = true;
        imgFrameIdxMap.resize(nrFramesGenMatches);
        for(size_t i = 0; i < nrFramesGenMatches; ++i){
            vector<pair<size_t,size_t>> imgUsageFrequ;
            map<size_t,size_t> imgIdx;
            imgUsageFrequ.emplace_back(make_pair(featureImgIdx[featureIdx], 1));
            imgIdx[featureImgIdx[featureIdx]] = 0;
            featureIdx++;
            size_t idx = 1;
            for (size_t j = 1; j < nrCorrs[i]; ++j) {
                if(imgIdx.find(featureImgIdx[featureIdx]) != imgIdx.end()){
                    imgUsageFrequ[imgIdx[featureImgIdx[featureIdx]]].second++;
                }else{
                    imgUsageFrequ.emplace_back(make_pair(featureImgIdx[featureIdx], 1));
                    imgIdx[featureImgIdx[featureIdx]] = idx;
                    idx++;
                }
                featureIdx++;
            }
            //Sort based on frequency of usage
            sort(imgUsageFrequ.begin(),
                    imgUsageFrequ.end(),
                    [](pair<size_t,size_t> &first, pair<size_t,size_t> &second){return first.second > second.second;});
            for(size_t j = 0; j < min(maxImgLoad, imgUsageFrequ.size()); ++j){
                imgFrameIdxMap[i].first[imgUsageFrequ[j].first] = j;
                imgFrameIdxMap[i].second.push_back(imgUsageFrequ[j].first);
            }
        }
    }

    return true;
}

void genMatchSequ::generateCorrespondingFeatures(){
    static size_t featureIdxBegin = 0;
    //Load images if not already done
    if(loadImgsEveryFrame){
        imgs.clear();
        imgs.reserve(imgFrameIdxMap[actFrameCnt].second.size());
        for(auto& i : imgFrameIdxMap[actFrameCnt].second){
            imgs.emplace_back(cv::imread(imageList[i], CV_LOAD_IMAGE_GRAYSCALE));
        }
    }
    size_t featureIdxBegin_tmp = featureIdxBegin;
    generateCorrespondingFeaturesTP(featureIdxBegin);
    featureIdxBegin_tmp += combNrCorrsTP;
    generateCorrespondingFeaturesTN(featureIdxBegin_tmp);
    featureIdxBegin += nrCorrs[actFrameCnt];
}

void genMatchSequ::generateCorrespondingFeaturesTP(size_t featureIdxBegin){
    //Generate feature for every TP
    int show_cnt = 0;
    size_t featureIdx = featureIdxBegin;
    for (int i = 0; i < combNrCorrsTP; ++i) {
        //Calculate homography
        Mat X = Mat(comb3DPts[i], true).reshape(1);
        bool visualize = false;
        if((verbose & SHOW_PLANES_FOR_HOMOGRAPHY) && ((show_cnt % 40) == 0)){
            visualize = true;
        }
        show_cnt++;
        Mat H = getHomographyForDistortionChkOld(X,
                                                 combCorrsImg1TP.col(i),
                                                 combCorrsImg2TP.col(i),
                                                 combCorrsImg12TP_IdxWorld[i],
                                                 featureIdx,
                                                 visualize);
        //Get image (check if already in memory)
        Mat img;
        if(loadImgsEveryFrame){
            if(imgFrameIdxMap[actFrameCnt].first.find(featureImgIdx[featureIdx]) != imgFrameIdxMap[actFrameCnt].first.end()){
                img = imgs[imgFrameIdxMap[actFrameCnt].first[featureImgIdx[featureIdx]]];
            }
            else{
                img = cv::imread(imageList[featureImgIdx[featureIdx]], CV_LOAD_IMAGE_GRAYSCALE);
            }
        } else{
            img = imgs[featureImgIdx[featureIdx]];
        }

        //Extract image patch
        KeyPoint kp = keypoints1[featureIdx];

        //Calculate the rotated ellipse from the keypoint size (circle) after applying the homography to the circle
        // to estimate the minimal necessary patch size
        cv::Rect patchROIimg1, patchROIimg2;
        cv::Point2d ellipseCenter;
        double ellipseRot = 0;
        cv::Size2d axes;
        bool useFallBack = false;
        bool noEllipse = false;
        if(!getRectFitsInEllipse(H, kp, patchROIimg1, patchROIimg2, ellipseCenter, ellipseRot, axes)){
            //If the calculation of the necessary patch size failed, calculate a standard patch
            noEllipse = true;
            const double minPatchSize = 41.0;
            do {
                useFallBack = false;
                int patchSize = minPatchSize2;//Must be an odd number
                int patchSize2 = (patchSize - 1) / 2;
                double width;
                double height;
                double minx, maxx, miny, maxy;
                Rect imgROI;
                do {
                    Point2i pt((int) round(kp.pt.x) - patchSize2, (int) round(kp.pt.y) - patchSize2);
                    imgROI = Rect(pt, Size(patchSize, patchSize));
                    if (imgROI.x < 0) {
                        imgROI.width += imgROI.x;
                        imgROI.x = 0;
                    } else if ((imgROI.x + imgROI.width) >= imgSize.width) {
                        imgROI.width = imgSize.width - imgROI.x;
                    }
                    if (imgROI.y < 0) {
                        imgROI.height += imgROI.y;
                        imgROI.y = 0;
                    } else if ((imgROI.y + imgROI.height) >= imgSize.height) {
                        imgROI.height = imgSize.height - imgROI.y;
                    }
                    //Check size after warping
                    Mat corners = Mat::ones(3, 4, CV_64FC1);
                    corners.col(0) = (Mat_<double>(3, 1) << (double) imgROI.x, (double) imgROI.y, 1.0);
                    corners.col(1) = (Mat_<double>(3, 1) << (double) (imgROI.x + imgROI.width), (double) imgROI.y, 1.0);
                    corners.col(2) = (Mat_<double>(3, 1) << (double) (imgROI.x + imgROI.width), (double) (imgROI.y +
                                                                                                          imgROI.height), 1.0);
                    corners.col(3) = (Mat_<double>(3, 1) << (double) imgROI.x, (double) (imgROI.y +
                                                                                         imgROI.height), 1.0);
                    Mat corners2 = Mat::ones(3, 4, CV_64FC1);
                    for (int j = 0; j < 4; ++j) {
                        corners2.col(j) = H * corners.col(j);
                        if (!isfinite(corners2.at<double>(0, j))
                            || !isfinite(corners2.at<double>(1, j))
                            || !isfinite(corners2.at<double>(2, j))) {
                            useFallBack = true;
                            break;
                        }
                        corners2.col(j) /= corners2.at<double>(2, j);
                        if (corners2.at<double>(0, j) < 0) {
                            corners2.at<double>(0, j) = 0;
                        } else if (corners2.at<double>(0, j) > (double) (imgSize.width - 1)) {
                            corners2.at<double>(0, j) = (double) (imgSize.width - 1);
                        }
                        if (corners2.at<double>(1, j) < 0) {
                            corners2.at<double>(1, j) = 0;
                        } else if (corners2.at<double>(1, j) > (double) (imgSize.height - 1)) {
                            corners2.at<double>(1, j) = (double) (imgSize.height - 1);
                        }
                    }
                    if (useFallBack)
                        break;
                    if (corners2.at<double>(0, 0) > corners2.at<double>(0, 1)) {//Reflection about y-axis
                        minx = max(corners2.at<double>(0, 1), corners2.at<double>(0, 2));
                        maxx = min(corners2.at<double>(0, 0), corners2.at<double>(0, 3));
                    } else {
                        minx = max(corners2.at<double>(0, 0), corners2.at<double>(0, 3));
                        maxx = min(corners2.at<double>(0, 1), corners2.at<double>(0, 2));
                    }
                    if (corners2.at<double>(1, 0) > corners2.at<double>(1, 3)) {//Reflection about x-axis
                        miny = max(corners2.at<double>(1, 3), corners2.at<double>(1, 2));
                        maxy = min(corners2.at<double>(1, 0), corners2.at<double>(1, 1));
                    } else {
                        miny = max(corners2.at<double>(1, 0), corners2.at<double>(1, 1));
                        maxy = min(corners2.at<double>(1, 3), corners2.at<double>(1, 2));
                    }
                    width = maxx - minx;
                    height = maxy - miny;
                    if ((width > (double) (maxPatchSizeMult2 * minPatchSize2))
                        || (height > (double) (maxPatchSizeMult2 * minPatchSize2))) {
                        useFallBack = true;
                        break;
                    }
                    int patchSize_tmp = (int) ceil(1.2f * (float) patchSize);//Must be an odd number
                    patchSize_tmp += (patchSize_tmp + 1) % 2;
                    int patchSize2_tmp = (patchSize_tmp - 1) / 2;
                    if ((width < minPatchSize) || (height < minPatchSize)) {
                        patchSize = patchSize_tmp;
                        patchSize2 = patchSize2_tmp;
                    }
                } while (((width < minPatchSize) || (height < minPatchSize)) && !useFallBack);
                if (!useFallBack) {
                    patchROIimg2 = Rect((int) ceil(minx - DBL_EPSILON),
                                        (int) ceil(miny - DBL_EPSILON),
                                        (int) floor(width + DBL_EPSILON),
                                        (int) floor(height + DBL_EPSILON));
                    patchROIimg1 = imgROI;
                } else {
                    //Construct a random affine homography
                    //Generate the non-isotropic scaling of the deformation (shear)
                    double d1, d2;
                    d1 = getRandDoubleValRng(0.8, 1.0, rand_gen);
                    d2 = getRandDoubleValRng(0.8, 1.0, rand_gen);
                    size_t sign1 = rand2() % 2;
                    for (int j = 0; j < 2; ++j) {//Higher propability to get a positive sign (sign1==0)
                        sign1 *= rand2() % 2;
                    }
                    if (sign1) {
                        d1 *= -1.0;
                    }
                    sign1 = rand2() % 2;
                    for (int j = 0; j < 2; ++j) {//Higher propability to get a positive sign (sign1==0)
                        sign1 *= rand2() % 2;
                    }
                    if (sign1) {
                        d2 *= -1.0;
                    }
                    Mat D = Mat::eye(2, 2, CV_64FC1);
                    D.at<double>(0, 0) = d1;
                    D.at<double>(1, 1) = d2;
                    //Generate a rotation for the deformation (shear)
                    double angle_rot = getRandDoubleValRng(0, M_PI_4, rand_gen);
                    Mat Rdeform = (Mat_<double>(2, 2) << std::cos(angle_rot), (-1. * std::sin(angle_rot)),
                            std::sin(angle_rot), std::cos(angle_rot));
                    //Generate a rotation
                    angle_rot = getRandDoubleValRng(0, M_PI_2, rand_gen);
                    Mat Rrot = (Mat_<double>(2, 2) << std::cos(angle_rot), (-1. * std::sin(angle_rot)),
                            std::sin(angle_rot), std::cos(angle_rot));
                    double scale = getRandDoubleValRng(0.65, 1.35, rand_gen);
                    //Calculate the new affine homography (without translation)
                    Mat Haff2 = scale * Rrot * Rdeform.t() * D * Rdeform;
                    H = Mat::eye(3, 3, CV_64FC1);
                    Haff2.copyTo(H.colRange(0, 2).rowRange(0, 2));
                }
            }while(useFallBack);
        }

        //Extract and warp the patch if the patch size is valid

        //Adapt H to eliminate wrong translation inside the patch
        //Translation to start at (0,0) in the warped image for the selected ROI arround the original image
        // (with coordinates in the original image based on the full image)
        Mat wiTo0 = Mat::eye(3,3,CV_64FC1);
        wiTo0.at<double>(0,2) = -patchROIimg2.x;
        wiTo0.at<double>(1,2) = -patchROIimg2.y;
        Mat H1 = wiTo0 * H;
        //Translation for the original image to ensure that points starting (left upper corner) at (0,0)
        // are mapped to the image ROI of the warped image
        Mat x2 = (Mat_<double>(3,1) << patchROIimg2.x, patchROIimg2.y, 1.0);
        Mat Hi = H.inv();
        Mat tm = Hi * x2;
        tm /= tm.at<double>(2);
        Mat tback = Mat::eye(3,3,CV_64FC1);
        tback.at<double>(0,2) = tm.at<double>(0);
        tback.at<double>(1,2) = tm.at<double>(1);
        Mat H2 = H1 * tback;
        Mat patchw;
        warpPerspective(img(patchROIimg1), patchw, H2, patchROIimg2.size(), INTER_LINEAR, BORDER_REPLICATE);

        //Show the patches
        if((verbose & SHOW_WARPED_PATCHES) && (((show_cnt - 1) % 40) == 0)){
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
                ellipseCenter1.x -= patchROIimg2.x;
                ellipseCenter1.y -= patchROIimg2.y;
                cv::Point c((int)round(ellipseCenter1.x), (int)round(ellipseCenter1.y));
                CV_Assert((ellipseCenter1.x >= 0) && (ellipseCenter1.y >= 0)
                && (ellipseCenter1.x < (double)patchROIimg2.width)
                && (ellipseCenter1.y < (double)patchROIimg2.height));
                cv::Size si((int)round(axes.width), (int)round(axes.height));
                Mat patchwc;
                cvtColor(patchw, patchwc, cv::COLOR_GRAY2BGR);
                cv::ellipse(patchwc, c, si, ellipseRot, 0, 360.0, Scalar(0,0,255));
                if(border2x || border2y) {
                    cv::copyMakeBorder(patchwc, patchwc, 0, border2y, 0, border2x, BORDER_CONSTANT, Scalar::all(0));
                }
                Mat patchc;
                cvtColor(img(patchROIimg1), patchc, cv::COLOR_GRAY2BGR);
                c = Point((int)round(kp.pt.x) - patchROIimg1.x, (int)round(kp.pt.y) - patchROIimg1.y);
                cv::circle(patchc, c, (int)round(kp.size / 2.f), Scalar(0,0,255));
                if(border1x || border1y) {
                    cv::copyMakeBorder(patchc, patchc, 0, border1y, 0, border1x, BORDER_CONSTANT, Scalar::all(0));
                }
                Mat bothPathes;
                cv::hconcat(patchc, patchwc, bothPathes);
                namedWindow("Original and warped patch with keypoint", WINDOW_AUTOSIZE);
                imshow("Original and warped patch with keypoint", bothPathes);

                waitKey(0);
                destroyWindow("Original and warped patch with keypoint");
            }
        }

        //Get the exact position of the keypoint in the patch
        Mat kpm = (Mat_<double>(3,1) << (double)kp.pt.x, (double)kp.pt.y, 1.0);
        kpm = H1 * kpm;
        kpm /= kpm.at<double>(2);
        Point2f ptm = Point2f((float)kpm.at<double>(0), (float)kpm.at<double>(1));

        //Check if we have to use a keypoint detector
        bool keypDetNeed = (noEllipse || !nearZero((double)kp.angle + 1.0) || (kp.octave != 0) || (kp.class_id != -1));
        cv::KeyPoint kp2;
        if(parsMtch.keypPosErrType || keypDetNeed){
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
                } else{
                    size_t j = 0;
                    for (; j < dists.size(); ++j) {
                        if(dists[0].second > 5.f){
                            break;
                        }
                    }
                    if(j > 0) {
                        //Calculate the overlap area between the found keypoints and the ellipse
                        cv::Point2d ellipseCenter1 = ellipseCenter;
                        ellipseCenter1.x -= patchROIimg2.x;
                        ellipseCenter1.y -= patchROIimg2.y;
                        cv::Point c((int) round(ellipseCenter1.x), (int) round(ellipseCenter1.y));
                        CV_Assert((ellipseCenter1.x >= 0) && (ellipseCenter1.y >= 0)
                                  && (ellipseCenter1.x < (double) patchROIimg2.width)
                                  && (ellipseCenter1.y < (double) patchROIimg2.height));
                        cv::Size si((int) round(axes.width), (int) round(axes.height));
                        Mat patchmask = Mat::zeros(patchw.size(), patchw.type());
                        cv::ellipse(patchmask, c, si, ellipseRot, 0, 360.0, Scalar::all(255), -1);
                        vector<size_t, int> overlapareas(j + 1);
                        for (size_t k = 0; k < overlapareas.size(); ++k) {
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
                        //Take the keypoint with the largest overlap
                        kp2 = kps2[overlapareas[0].first];
                    }else{
                        kp2 = kps2[dists[0].first];
                    }
                }
                if(!parsMtch.keypPosErrType){
                    //Correct the keypoint position to the exact location
                    kp2.pt = ptm;
                }
            }
        } else if(!noEllipse){
            //Use the dimension of the ellipse to get the scale/size of the keypoint
            kp2 = kp;
            kp2.pt = ptm;
            kp2.size = (float)axes.width;
        }

        //Calculate the descriptor
        if(!useFallBack){
            //Apply noise



            vector<KeyPoint> pkp21(1);
            pkp21[0] = kp2;
            Mat descr21;
            if (matchinglib::getDescriptors(img,
                                            pkp21,
                                            parsMtch.descriptorType,
                                            descr21,
                                            parsMtch.keyPointType) != 0) {
                useFallBack = true;
            }else{
                //Check matchability
            }
        }

        if(useFallBack){
            //Only add gaussian noise and salt and pepper noise to the original patch
        }




        featureIdx++;
    }
}

bool genMatchSequ::getRectFitsInEllipse(const cv::Mat &H,
        const cv::KeyPoint &kp,
        cv::Rect &patchROIimg1,
        cv::Rect &patchROIimg2,
        cv::Point2d &ellipseCenter,
        double &ellipseRot,
        cv::Size2d &axes){
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
        teta = std::atan(QH.at<double>(0,1) / (QH.at<double>(0,0) - QH.at<double>(1,1))) / 2.0;
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

    ellipseCenter = Point2d(x0, y0);

    //Get division coefficients a^2 and b^2 of backrotated ellipse
    //(x' - x0')^2 / a^2 + (y' - y0')^2 / b^2 = 1
    double dom = -4.0 * F1 * A1 * C1 + C1 * D1 * D1 + A1 * E1 * E1;
    double a2 = dom / (4.0 * A1 * C1 * C1);//Corresponds to half length of major axis
    double b2 = dom / (4.0 * A1 * A1 * C1);//Corresponds to half length of minor axis
    if(!isfinite(a2) || !isfinite(b2)){
        return false;
    }

    axes = Size2d(a2, b2);

    //Calculate 2D vector of major and minor axis
    Mat ma = (Mat_<double>(2,1) << a2 * cosTeta, a2 * sinTeta);
    Mat mi = (Mat_<double>(2,1) << -b2 * sinTeta, b2 * cosTeta);

    //Calculate corner points of rotated rectangle enclosing ellipse
    Mat corners = Mat(2,4,CV_64FC1);
    corners.col(0) = ma + mi;
    corners.col(0) = ma - mi;
    corners.col(0) = -1.0 * ma + mi;
    corners.col(0) = -1.0 * ma - mi;

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
        minSquare = minPatchSize2;
    }
    double minSquare2 = (double)((minSquare - 1) / 2);


    //Transform the square back into the original image
    Mat corners1 = Mat::ones(3,4,CV_64FC1);
    x0 = round(x0);
    y0 = round(y0);
    corners1.col(0) = (Mat_<double>(3,1) << x0 - minSquare2, y0 - minSquare2, 1.0);
    corners1.col(1) = (Mat_<double>(3,1) << x0 + minSquare2, y0 - minSquare2, 1.0);
    corners1.col(2) = (Mat_<double>(3,1) << x0 + minSquare2, y0 + minSquare2, 1.0);
    corners1.col(3) = (Mat_<double>(3,1) << x0 - minSquare2, y0 + minSquare2, 1.0);
    patchROIimg2 = cv::Rect((int)floor(x0 - minSquare2 + DBL_EPSILON),
            (int)floor(y0 - minSquare2 + DBL_EPSILON),
            minSquare,
            minSquare);
    Mat corners2 = Mat::ones(3,4,CV_64FC1);
    bool atBorder = false;
    for (int j = 0; j < 4; ++j) {
        corners2.col(j) = Hi * corners1.col(j);
        corners2.col(j) /= corners2.at<double>(2,j);
        if(corners2.at<double>(0,j) < 0){
            corners2.at<double>(0,j) = 0;
            atBorder = true;
        }else if(corners2.at<double>(0,j) > (double)(imgSize.width - 1)){
            corners2.at<double>(0,j) = (double)(imgSize.width - 1);
            atBorder = true;
        }
        if(corners2.at<double>(1,j) < 0){
            corners2.at<double>(1,j) = 0;
            atBorder = true;
        }else if(corners2.at<double>(1,j) > (double)(imgSize.height - 1)){
            corners2.at<double>(1,j) = (double)(imgSize.height - 1);
            atBorder = true;
        }
    }
    //Adapt the ROI in the transformed image if we are at the border
    if(atBorder){
        for (int j = 0; j < 4; ++j) {
            corners1.col(j) = H * corners2.col(j);
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
    patchROIimg1 = cv::Rect((int)ceil(minx - DBL_EPSILON),
                            (int)ceil(miny - DBL_EPSILON),
                            (int)floor(dimx + DBL_EPSILON),
                            (int)floor(dimy + DBL_EPSILON));

    return true;
}

void genMatchSequ::generateCorrespondingFeaturesTN(size_t featureIdxBegin){

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
    ss << parsMtch.lostCorrPor;
    ss << parsMtch.mainStorePath;
    ss << parsMtch.storePtClouds;
    ss << parsMtch.takeLessFramesIfLessKeyP;

    strFromPars = ss.str();

    std::hash<std::string> hash_fn;
    return hash_fn(strFromPars);
}

bool genMatchSequ::writeMatchingParameters(){
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
    string filename = concatPath(sequParPath, matchInfoFName);
    FileStorage fs;
    if(checkFileExists(filename)){
        fs = FileStorage(filename, FileStorage::APPEND);
        if (!fs.isOpened()) {
            cerr << "Failed to open " << filename << endl;
            return false;
        }
        cvWriteComment(*fs, "\n\nNext parameters:\n", 0);
    }
    else{
        fs = FileStorage(filename, FileStorage::WRITE);
        if (!fs.isOpened()) {
            cerr << "Failed to open " << filename << endl;
            return false;
        }
        cvWriteComment(*fs, "This file contains the directory name and its corresponding parameters for "
                            "generating matches out of given 3D correspondences.\n\n", 0);
    }

    cvWriteComment(*fs, "Directory name (within the path containing this file) which holds matching results "
                        "using the below parameters.", 0);
    size_t posLastSl = matchDataPath.rfind('/');
    string matchDirName = matchDataPath.substr(posLastSl + 1);
    fs << "hashMatchingPars" << matchDirName;

    cvWriteComment(*fs, "Path containing the images for producing keypoint patches", 0);
    fs << "imgPath" << parsMtch.imgPath;
    cvWriteComment(*fs, "Image pre- and/or postfix for images within imgPath", 0);
    fs << "imgPrePostFix" << parsMtch.imgPrePostFix;
    cvWriteComment(*fs, "Name of keypoint detector", 0);
    fs << "keyPointType" << parsMtch.keyPointType;
    cvWriteComment(*fs, "Name of descriptor extractor", 0);
    fs << "descriptorType" << parsMtch.descriptorType;
    cvWriteComment(*fs, "Keypoint detector error (true) or error normal distribution (false)", 0);
    fs << "keypPosErrType" << parsMtch.keypPosErrType;
    cvWriteComment(*fs, "Keypoint error distribution (mean, std)", 0);
    fs << "keypErrDistr";
    fs << "{" << "first" << parsMtch.keypErrDistr.first;
    fs << "second" << parsMtch.keypErrDistr.second << "}";
    cvWriteComment(*fs, "Noise (mean, std) on the image intensity for descriptor calculation", 0);
    fs << "imgIntNoise";
    fs << "{" << "first" << parsMtch.imgIntNoise.first;
    fs << "second" << parsMtch.imgIntNoise.second << "}";
    cvWriteComment(*fs, "Portion (0 to 0.9) of lost correspondences from frame to frame.", 0);
    fs << "lostCorrPor" << parsMtch.lostCorrPor;
    cvWriteComment(*fs, "If true, all PCL point clouds and necessary information to load a cam sequence "
                        "with correspondences are stored to disk", 0);
    fs << "storePtClouds" << parsMtch.storePtClouds;
    cvWriteComment(*fs, "If true, the parameters and information are stored and read in XML format.", 0);
    fs << "rwXMLinfo" << parsMtch.rwXMLinfo;
    cvWriteComment(*fs, "If true, the stored information and parameters are compressed", 0);
    fs << "compressedWrittenInfo" << parsMtch.compressedWrittenInfo;
    cvWriteComment(*fs, "If true and too less images images are provided (resulting in too less keypoints), "
                        "only as many frames with GT matches are provided as keypoints are available.", 0);
    fs << "takeLessFramesIfLessKeyP" << parsMtch.takeLessFramesIfLessKeyP;

    return true;
}

bool genMatchSequ::writeSequenceOverviewPars(){
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
    string filename = concatPath(parsMtch.mainStorePath, matchInfoFName);
    FileStorage fs;
    if(checkFileExists(filename)){
        fs = FileStorage(filename, FileStorage::APPEND);
        if (!fs.isOpened()) {
            cerr << "Failed to open " << filename << endl;
            return false;
        }
        cvWriteComment(*fs, "\n\nNext parameters:\n", 0);
    }
    else{
        fs = FileStorage(filename, FileStorage::WRITE);
        if (!fs.isOpened()) {
            cerr << "Failed to open " << filename << endl;
            return false;
        }
        cvWriteComment(*fs, "This file contains the directory name and its corresponding parameters for "
                            "generating 3D correspondences.\n\n", 0);
    }

    cvWriteComment(*fs, "Directory name (within the path containing this file) which holds multiple frames of "
                        "3D correspondences using the below parameters.", 0);
    size_t posLastSl = sequParPath.rfind('/');
    string sequDirName = sequParPath.substr(posLastSl + 1);
    fs << "hashMatchingPars" << sequDirName;

    writeSomeSequenceParameters(fs);

    return true;
}

static inline FileStorage& operator << (FileStorage& fs, bool &value)
{
    if(value){
        return (fs << 1);
    }

    return (fs << 0);
}

static inline FileStorage& operator << (FileStorage& fs, int64_t &value)
{
    string strVal = std::to_string(value);
    return (fs << strVal);
}

static inline void operator >> (const FileNode& n, int64_t& value)
{
    string strVal;
    n >> strVal;
    value = std::stoll(strVal);
}

static inline FileNodeIterator& operator >> (FileNodeIterator& it, int64_t & value)
{
    *it >> value;
    return ++it;
}

void genMatchSequ::writeSomeSequenceParameters(cv::FileStorage &fs){
    cvWriteComment(*fs, "Number of different stereo camera configurations", 0);
    fs << "nrStereoConfs" << (int) nrStereoConfs;
    cvWriteComment(*fs, "Different rotations between stereo cameras", 0);
    fs << "R" << "[";
    for (auto &i : R) {
        fs << i;
    }
    fs << "]";
    cvWriteComment(*fs, "Different translation vectors between stereo cameras", 0);
    fs << "t" << "[";
    for (auto &i : t) {
        fs << i;
    }
    fs << "]";
    cvWriteComment(*fs, "Inlier ratio for every frame", 0);
    fs << "inlRat" << "[";
    for (auto &i : inlRat) {
        fs << i;
    }
    fs << "]";
    cvWriteComment(*fs, "# of Frames per camera configuration", 0);
    fs << "nFramesPerCamConf" << (int) pars.nFramesPerCamConf;
    cvWriteComment(*fs, "Total number of frames in the sequence", 0);
    fs << "totalNrFrames" << (int) totalNrFrames;
    cvWriteComment(*fs, "Absolute number of correspondences (TP+TN) per frame", 0);
    fs << "nrCorrs" << "[";
    for (auto &i : nrCorrs) {
        fs << (int) i;
    }
    fs << "]";
    cvWriteComment(*fs, "portion of correspondences at depths", 0);
    fs << "corrsPerDepth";
    fs << "{" << "near" << pars.corrsPerDepth.near;
    fs << "mid" << pars.corrsPerDepth.mid;
    fs << "far" << pars.corrsPerDepth.far << "}";
    cvWriteComment(*fs, "List of portions of image correspondences at regions", 0);
    fs << "corrsPerRegion" << "[";
    for (auto &i : pars.corrsPerRegion) {
        fs << i;
    }
    fs << "]";
    cvWriteComment(*fs, "Number of moving objects in the scene", 0);
    fs << "nrMovObjs" << (int) pars.nrMovObjs;
    cvWriteComment(*fs, "Relative area range of moving objects", 0);
    fs << "relAreaRangeMovObjs";
    fs << "{" << "first" << pars.relAreaRangeMovObjs.first;
    fs << "second" << pars.relAreaRangeMovObjs.second << "}";
    cvWriteComment(*fs, "Depth of moving objects.", 0);
    fs << "movObjDepth" << "[";
    for (auto &i : pars.movObjDepth) {
        fs << (int) i;
    }
    fs << "]";
    cvWriteComment(*fs,
                   "Absolute coordinates of the camera centres (left or bottom cam of stereo rig) for every frame.", 0);
    fs << "absCamCoordinates" << "[";
    for (auto &i : absCamCoordinates) {
        fs << "{" << "R" << i.R;
        fs << "t" << i.t << "}";
    }
    fs << "]";
}

bool genMatchSequ::writeSequenceParameters(const std::string &filename) {
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    cvWriteComment(*fs, "This file contains all parameters used to generate "
                        "multiple consecutive frames with stereo correspondences.\n", 0);

    cvWriteComment(*fs, "# of Frames per camera configuration", 0);
    fs << "nFramesPerCamConf" << (int) pars.nFramesPerCamConf;
    cvWriteComment(*fs, "Inlier ratio range", 0);
    fs << "inlRatRange";
    fs << "{" << "first" << pars.inlRatRange.first;
    fs << "second" << pars.inlRatRange.second << "}";
    cvWriteComment(*fs, "Inlier ratio change rate from pair to pair", 0);
    fs << "inlRatChanges" << pars.inlRatChanges;
    cvWriteComment(*fs, "# true positives range", 0);
    fs << "truePosRange";
    fs << "{" << "first" << (int) pars.truePosRange.first;
    fs << "second" << (int) pars.truePosRange.second << "}";
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
    for (auto &i : pars.corrsPerRegion) {
        fs << i;
    }
    fs << "]";
    cvWriteComment(*fs, "Repeat rate of portion of correspondences at regions.", 0);
    fs << "corrsPerRegRepRate" << (int) pars.corrsPerRegRepRate;
    cvWriteComment(*fs, "Portion of depths per region", 0);
    fs << "depthsPerRegion" << "[";
    for (auto &i : pars.depthsPerRegion) {
        for (auto &j : i) {
            fs << "{" << "near" << j.near;
            fs << "mid" << j.mid;
            fs << "far" << j.far << "}";
        }
    }
    fs << "]";
    cvWriteComment(*fs, "Min and Max number of connected depth areas per region", 0);
    fs << "nrDepthAreasPReg" << "[";
    for (auto &i : pars.nrDepthAreasPReg) {
        for (auto &j : i) {
            fs << "{" << "first" << (int) j.first;
            fs << "second" << (int) j.second << "}";
        }
    }
    fs << "]";
    cvWriteComment(*fs, "Movement direction or track of the cameras", 0);
    fs << "camTrack" << "[";
    for (auto &i : pars.camTrack) {
        fs << i;
    }
    fs << "]";
    cvWriteComment(*fs, "Relative velocity of the camera movement", 0);
    fs << "relCamVelocity" << pars.relCamVelocity;
    cvWriteComment(*fs, "Rotation matrix of the first camera centre", 0);
    fs << "R" << pars.R;
    cvWriteComment(*fs, "Number of moving objects in the scene", 0);
    fs << "nrMovObjs" << (int) pars.nrMovObjs;
    cvWriteComment(*fs, "Possible starting positions of moving objects in the image", 0);
    fs << "startPosMovObjs" << pars.startPosMovObjs;
    cvWriteComment(*fs, "Relative area range of moving objects", 0);
    fs << "relAreaRangeMovObjs";
    fs << "{" << "first" << pars.relAreaRangeMovObjs.first;
    fs << "second" << pars.relAreaRangeMovObjs.second << "}";
    cvWriteComment(*fs, "Depth of moving objects.", 0);
    fs << "movObjDepth" << "[";
    for (auto &i : pars.movObjDepth) {
        fs << (int) i;
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
    fs << "minNrMovObjs" << (int) pars.minNrMovObjs;

    cvWriteComment(*fs, "Total number of frames in the sequence", 0);
    fs << "totalNrFrames" << (int) totalNrFrames;
    cvWriteComment(*fs, "Absolute number of correspondences (TP+TN) per frame", 0);
    fs << "nrCorrs" << "[";
    for (auto &i : nrCorrs) {
        fs << (int) i;
    }
    fs << "]";
    cvWriteComment(*fs,
                   "Absolute coordinates of the camera centres (left or bottom cam of stereo rig) for every frame.", 0);
    fs << "absCamCoordinates" << "[";
    for (auto &i : absCamCoordinates) {
        fs << "{" << "R" << i.R;
        fs << "t" << i.t << "}";
    }
    fs << "]";

    nrMovObjAllFrames = movObj3DPtsWorldAllFrames.size();
    cvWriteComment(*fs, "Number of moving object point clouds over all frames.", 0);
    fs << "nrMovObjAllFrames" << (int) nrMovObjAllFrames;

    //Write camera parameters
    cvWriteComment(*fs, "Camera matrix of cam 1", 0);
    fs << "K1" << K1;
    cvWriteComment(*fs, "Camera matrix of cam 2", 0);
    fs << "K2" << K2;
    cvWriteComment(*fs, "Image size", 0);
    fs << "imgSize";
    fs << "{" << "width" << imgSize.width;
    fs << "height" << imgSize.height << "}";

    fs.release();

    return true;
}

bool genMatchSequ::readSequenceParameters(const std::string &filename) {
    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    int tmp;
    fs["nFramesPerCamConf"] >> tmp;
    pars3D.nFramesPerCamConf = (size_t) tmp;

    FileNode n = fs["inlRatRange"];
    double first_dbl, second_dbl;
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    pars3D.inlRatRange = make_pair(first_dbl, second_dbl);

    fs["inlRatChanges"] >> pars3D.inlRatChanges;

    n = fs["truePosRange"];
    int first_int, second_int;
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
    if (n.type() != FileNode::SEQ) {
        cerr << "corrsPerRegion is not a sequence! FAIL" << endl;
        return false;
    }
    pars3D.corrsPerRegion.clear();
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        Mat m;
        it >> m;
        pars3D.corrsPerRegion.push_back(m.clone());
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
    if (n.type() != FileNode::SEQ) {
        cerr << "camTrack is not a sequence! FAIL" << endl;
        return false;
    }
    pars3D.camTrack.clear();
    it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        Mat m;
        it >> m;
        pars3D.camTrack.emplace_back(m.clone());
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
    if (n.type() != FileNode::SEQ) {
        cerr << "camTrack is not a sequence! FAIL" << endl;
        return false;
    }
    pars3D.movObjDepth.clear();
    it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        it >> tmp;
        pars3D.movObjDepth.push_back((depthClass) tmp);
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

    fs["totalNrFrames"] >> tmp;
    totalNrFrames = (size_t) tmp;

    n = fs["nrCorrs"];
    if (n.type() != FileNode::SEQ) {
        cerr << "nrCorrs is not a sequence! FAIL" << endl;
        return false;
    }
    nrCorrs.clear();
    it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        it >> tmp;
        nrCorrs.push_back((size_t) tmp);
    }

    n = fs["absCamCoordinates"];
    if (n.type() != FileNode::SEQ) {
        cerr << "absCamCoordinates is not a sequence! FAIL" << endl;
        return false;
    }
    absCamCoordinates.clear();
    it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        FileNode n1 = *it;
        Mat m1, m2;
        n1["R"] >> m1;
        n1["t"] >> m2;
        absCamCoordinates.emplace_back(Poses(m1.clone(), m2.clone()));
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

    cvWriteComment(*fs, "This file contains all correspondences of a single frame.\n", 0);

    cvWriteComment(*fs, "Frame number", 0);
    fs << "actFrameCnt" << (int) actFrameCnt;

    cvWriteComment(*fs, "Actual rotation matrix of the stereo rig: x2 = actR * x1 + actT", 0);
    fs << "actR" << actR;

    cvWriteComment(*fs, "Actual translation vector of the stereo rig: x2 = actR * x1 + actT", 0);
    fs << "actT" << actT;

    fs << "actDepthNear" << actDepthNear;
    fs << "actDepthMid" << actDepthMid;
    fs << "actDepthFar" << actDepthFar;

    cvWriteComment(*fs, "Combined TP correspondences (static and moving objects) of camera 1", 0);
    fs << "combCorrsImg1TP" << combCorrsImg1TP;
    cvWriteComment(*fs, "Combined TP correspondences (static and moving objects) of camera 2", 0);
    fs << "combCorrsImg2TP" << combCorrsImg2TP;

    cvWriteComment(*fs, "Combined 3D points corresponding to matches combCorrsImg1TP and combCorrsImg2TP", 0);
    fs << "comb3DPts" << "[";
    for (auto &i : comb3DPts) {
        fs << i;
    }
    fs << "]";

    /*cvWriteComment(*fs, "Index to the corresponding world 3D point within staticWorld3DPts and movObj3DPtsWorld of "
                        "combined TP correspondences (static and moving objects) in combCorrsImg1TP and "
                        "combCorrsImg2TP. Contains only the most 32 significant bits of the int64 indices.", 0);
    fs << "combCorrsImg12TP_IdxWorld_m32bit" << "[";
    for (auto &i : combCorrsImg12TP_IdxWorld) {
        int64_t mostsig = i >> 32;
        fs << (int32_t) mostsig;
    }
    fs << "]";
    cvWriteComment(*fs, "Index to the corresponding world 3D point within staticWorld3DPts and movObj3DPtsWorld of "
                        "combined TP correspondences (static and moving objects) in combCorrsImg1TP and "
                        "combCorrsImg2TP. Contains only the least 32 significant bits of the int64 indices.", 0);
    fs << "combCorrsImg12TP_IdxWorld_l32bit" << "[";
    for (auto &i : combCorrsImg12TP_IdxWorld) {
        int64_t leastsig = (i << 32) >> 32;
        fs << (int32_t) leastsig;
    }
    fs << "]";*/

    cvWriteComment(*fs, "Index to the corresponding world 3D point within staticWorld3DPts and movObj3DPtsWorld of "
                        "combined TP correspondences (static and moving objects) in combCorrsImg1TP and "
                        "combCorrsImg2TP.", 0);
    fs << "combCorrsImg12TP_IdxWorld" << "[";
    for (auto &i : combCorrsImg12TP_IdxWorld) {
        fs << i;
    }
    fs << "]";

    /*cvWriteComment(*fs, "Similar to combCorrsImg12TP_IdxWorld but the vector indices for moving objects do NOT "
                        "correspond with vector elements in movObj3DPtsWorld but with a consecutive number "
                        "pointing to moving object pointclouds that were saved after they emerged. "
                        "Contains only the most 32 significant bits of the int64 indices.", 0);
    fs << "combCorrsImg12TPContMovObj_IdxWorld_m32bit" << "[";
    for (auto &i : combCorrsImg12TPContMovObj_IdxWorld) {
        int64_t mostsig = i >> 32;
        fs << (int32_t) mostsig;
    }
    fs << "]";
    cvWriteComment(*fs, "Similar to combCorrsImg12TP_IdxWorld but the vector indices for moving objects do NOT "
                        "correspond with vector elements in movObj3DPtsWorld but with a consecutive number "
                        "pointing to moving object pointclouds that were saved after they emerged. "
                        "Contains only the least 32 significant bits of the int64 indices.", 0);
    fs << "combCorrsImg12TPContMovObj_IdxWorld_l32bit" << "[";
    for (auto &i : combCorrsImg12TPContMovObj_IdxWorld) {
        int64_t leastsig = (i << 32) >> 32;
        fs << (int32_t) leastsig;
    }
    fs << "]";*/

    cvWriteComment(*fs, "Similar to combCorrsImg12TP_IdxWorld but the vector indices for moving objects do NOT "
                        "correspond with vector elements in movObj3DPtsWorld but with a consecutive number "
                        "pointing to moving object pointclouds that were saved after they emerged.", 0);
    fs << "combCorrsImg12TPContMovObj_IdxWorld" << "[";
    for (auto &i : combCorrsImg12TPContMovObj_IdxWorld) {
        fs << i;
    }
    fs << "]";

    cvWriteComment(*fs, "Combined TN correspondences (static and moving objects) in camera 1", 0);
    fs << "combCorrsImg1TN" << combCorrsImg1TN;
    cvWriteComment(*fs, "Combined TN correspondences (static and moving objects) in camera 2", 0);
    fs << "combCorrsImg2TN" << combCorrsImg2TN;

    cvWriteComment(*fs, "Number of overall TP correspondences (static and moving objects)", 0);
    fs << "combNrCorrsTP" << combNrCorrsTP;
    cvWriteComment(*fs, "Number of overall TN correspondences (static and moving objects)", 0);
    fs << "combNrCorrsTN" << combNrCorrsTN;

    cvWriteComment(*fs, "Distance values of all (static and moving objects) TN keypoint locations in the 2nd "
                        "image to the location that would be a perfect correspondence to the TN in image 1.", 0);
    fs << "combDistTNtoReal" << "[";
    for (auto &i : combDistTNtoReal) {
        fs << i;
    }
    fs << "]";

    cvWriteComment(*fs, "Final number of new generated TP correspondences for static objects.", 0);
    fs << "finalNrTPStatCorrs" << finalNrTPStatCorrs;

    cvWriteComment(*fs, "Final number of new generated TP correspondences for moving objects.", 0);
    fs << "finalNrTPMovCorrs" << finalNrTPMovCorrs;

    cvWriteComment(*fs, "Final number of backprojected TP correspondences for static objects.", 0);
    fs << "finalNrTPStatCorrsFromLast" << finalNrTPStatCorrsFromLast;

    cvWriteComment(*fs, "Final number of backprojected TP correspondences for moving objects.", 0);
    fs << "finalNrTPMovCorrsFromLast" << finalNrTPMovCorrsFromLast;

    cvWriteComment(*fs, "Final number of TN correspondences for static objects.", 0);
    fs << "finalNrTNStatCorrs" << finalNrTNStatCorrs;

    cvWriteComment(*fs, "Final number of TN correspondences for moving objects.", 0);
    fs << "finalNrTNMovCorrs" << finalNrTNMovCorrs;

    cvWriteComment(*fs, "Order of correspondences in combined Mat combCorrsImg1TP, combCorrsImg2TP, and comb3DPts", 0);
    fs << "combCorrsImg12TPorder";
    fs << "{" << "statTPfromLast" << (int) combCorrsImg12TPorder.statTPfromLast;
    fs << "statTPnew" << (int) combCorrsImg12TPorder.statTPnew;
    fs << "movTPfromLast" << (int) combCorrsImg12TPorder.movTPfromLast;
    fs << "movTPnew" << (int) combCorrsImg12TPorder.movTPnew << "}";

    cvWriteComment(*fs, "Indicates that TN correspondences of static objects are located at the beginning of Mats "
                        "combCorrsImg1TN and combCorrsImg2TN", 0);
    /*if (combCorrsImg12TPstatFirst)
        fs << "finalNrTNMovCorrs" << 1;
    else
        fs << "finalNrTNMovCorrs" << 0;*/
    fs << "finalNrTNMovCorrs" << combCorrsImg12TPstatFirst;

    return true;
}

bool genMatchSequ::read3DInfoSingleFrame(const std::string &filename) {
    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    int tmp;

    fs["actFrameCnt"] >> tmp;
    actFrameCnt = (size_t) tmp;

    fs["actR"] >> actR;

    fs["actT"] >> actT;

    fs["actDepthNear"] >> actDepthNear;
    fs["actDepthMid"] >> actDepthMid;
    fs["actDepthFar"] >> actDepthFar;

    fs["combCorrsImg1TP"] >> combCorrsImg1TP;
    fs["combCorrsImg2TP"] >> combCorrsImg2TP;

    FileNode n = fs["comb3DPts"];
    if (n.type() != FileNode::SEQ) {
        cerr << "comb3DPts is not a sequence! FAIL" << endl;
        return false;
    }
    comb3DPts.clear();
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        cv::Point3d pt;
        it >> pt;
        comb3DPts.push_back(pt);
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
    if (n.type() != FileNode::SEQ) {
        cerr << "combCorrsImg12TP_IdxWorld is not a sequence! FAIL" << endl;
        return false;
    }
    combCorrsImg12TP_IdxWorld.clear();
    it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        int64_t val;
        it >> val;
        combCorrsImg12TP_IdxWorld.push_back(val);
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
    if (n.type() != FileNode::SEQ) {
        cerr << "combCorrsImg12TPContMovObj_IdxWorld is not a sequence! FAIL" << endl;
        return false;
    }
    combCorrsImg12TPContMovObj_IdxWorld.clear();
    it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        int64_t val;
        it >> val;
        combCorrsImg12TPContMovObj_IdxWorld.push_back(val);
    }

    fs["combCorrsImg1TN"] >> combCorrsImg1TN;
    fs["combCorrsImg2TN"] >> combCorrsImg2TN;

    fs["combNrCorrsTP"] >> combNrCorrsTP;
    fs["combNrCorrsTN"] >> combNrCorrsTN;

    n = fs["combDistTNtoReal"];
    if (n.type() != FileNode::SEQ) {
        cerr << "combDistTNtoReal is not a sequence! FAIL" << endl;
        return false;
    }
    combDistTNtoReal.clear();
    it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        double dist;
        it >> dist;
        combDistTNtoReal.push_back(dist);
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

    fs["finalNrTNMovCorrs"] >> tmp;
    combCorrsImg12TPstatFirst = (tmp != 0);

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
        if(checkOverwriteDelFiles(fname, "Output file for moving 3D PCL point cloud already exists:", overwrite)){
            return false;
        }
        pcl::io::savePCDFileBinaryCompressed(staticWorld3DPtsFileName, movObj3DPtsWorldAllFrames[i]);
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
            if (pcl::io::loadPCDFile(staticWorld3DPtsFileName, movObj3DPtsWorldAllFrames[i]) == -1) {
                cerr << "Could not read PCL point cloud " << staticWorld3DPtsFileName << endl;
                return false;
            }
        }
    }

    return true;
}
