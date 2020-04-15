/**********************************************************************************************************
FILE: generateSequence.cpp

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: March 2018

LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for generating stereo sequences with correspondences given
a view restrictions like depth ranges, moving objects, ...
**********************************************************************************************************/

#include "generateSequence.h"
#include "opencv2/imgproc/imgproc.hpp"
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#include <array>
#include <chrono>

#include "pcl/filters/frustum_culling.h"
#include "pcl/common/transforms.h"
#include "pcl/filters/voxel_grid_occlusion_estimation.h"
#include <pcl/common/common.h>

#include <boost/thread/thread.hpp>
//#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "nanoflann_utils.h"
#include <nanoflann.hpp>

#include "polygon_helper.h"

#include "side_funcs.h"
#include "io_data.h"


using namespace std;
using namespace cv;

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */



/* -------------------------- Functions -------------------------- */

genStereoSequ::genStereoSequ(cv::Size imgSize_,
                             cv::Mat K1_,
                             cv::Mat K2_,
                             std::vector<cv::Mat> R_,
                             std::vector<cv::Mat> t_,
                             StereoSequParameters &pars_,
                             bool filter_occluded_points_,
                             uint32_t verbose_,
                             const std::string &writeIntermRes_path_) :
        verbose(verbose_),
        writeIntermRes_path(writeIntermRes_path_),
        filter_occluded_points(filter_occluded_points_),
        imgSize(imgSize_),
        K1(K1_),
        K2(K2_),
        pars(pars_),
        R(std::move(R_)),
        t(std::move(t_)) {
    CV_Assert((K1.rows == 3) && (K2.rows == 3) && (K1.cols == 3) && (K2.cols == 3) && (K1.type() == CV_64FC1) &&
              (K2.type() == CV_64FC1));
    CV_Assert((imgSize.area() > 0) && (R.size() == t.size()) && !R.empty());

    chrono::high_resolution_clock::time_point t1, t2;
    t1 = chrono::high_resolution_clock::now();

    long int seed = randSeed(rand_gen);
    randSeed(rand2, seed);

    //Generate a mask with the minimal distance between keypoints and a mask for marking used areas in the first stereo image
    genMasks();

    //Calculate inverse of camera matrices
    K1i = K1.inv();
    K2i = K2.inv();

    //Calculate distorted camera matrices (only used for output)
    calcDistortedIntrinsics();

    //Number of stereo configurations
    nrStereoConfs = R.size();

    //Construct the camera path
    constructCamPath();

    //Calculate the thresholds for the depths near, mid, and far for every camera configuration
    if (!getDepthRanges()) {
        throw SequenceException("Depth ranges are negative!\n");
    }

    //Used inlier ratios
    genInlierRatios();

    //Initialize region ROIs and masks
    genRegMasks();

    //Number of correspondences per image and Correspondences per image regions
    initNrCorrespondences();

    //Depths per image region
    adaptDepthsPerRegion();

    //Check if the given ranges of connected depth areas per image region are correct and initialize them for every definition of depths per image region
    checkDepthAreas();

    //Calculate the area in pixels for every depth and region
    calcPixAreaPerDepth();

    //Reset variables for the moving objects & Initialize variable for storing static 3D world coordinates
    resetInitVars();

    //Calculate the initial number, size, and positions of moving objects in the image
    getNrSizePosMovObj();

    //Get the relative movement direction (compared to the camera movement) for every moving object
    checkMovObjDirection();

    //Calculate the execution time
    t2 = chrono::high_resolution_clock::now();
    tus_to_init = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
}

void genStereoSequ::resetInitVars(){
    //Reset variables for the moving objects
    combMovObjLabelsAll = Mat::zeros(imgSize, CV_8UC1);
    movObjMask2All = Mat::zeros(imgSize, CV_8UC1);
    movObjMaskFromLast = Mat::zeros(imgSize, CV_8UC1);
    movObjMaskFromLast2 = Mat::zeros(imgSize, CV_8UC1);
    movObjHasArea = std::vector<std::vector<bool>>(3, std::vector<bool>(3, false));
    actCorrsOnMovObj = 0;
    actCorrsOnMovObjFromLast = 0;
    movObj3DPtsWorldAllFrames.clear();

    //Initialize variable for storing static 3D world coordinates
    staticWorld3DPts.reset(new pcl::PointCloud<pcl::PointXYZ>());
}

void genStereoSequ::genMasks() {
    //Generate a mask with the minimal distance between keypoints
    int sqrSi = 2 * max((int) ceil(pars.minKeypDist), 1) + 1;
    csurr = Mat::ones(sqrSi, sqrSi, CV_8UC1);

    //Generate a mask for marking used areas in the first stereo image
    corrsIMG = Mat::zeros(imgSize.height + sqrSi - 1, imgSize.width + sqrSi - 1, CV_8UC1);

    //Calculate the average area a keypoint occupies if it is masked by csurr taking into account the probabilies of
    //different mask overlaps
    calcAvgMaskingArea();
}

void genStereoSequ::calcAvgMaskingArea(){
    int csurrArea = csurr.rows * csurr.cols;
    avgMaskingArea = 0;
    double tmp = 0;
    /*avg. A = Sum(i^2)/Sum(i) with i = [1, 2, ...,d^2] and d the edge length of the mask used for reserving the
     * minimal distance for keypoints. i^2 is used as a bigger free area has a higher probability (proportional to
     * its area) to be hit by a random generator to generate a new keypoint e.g. a non-masked area of 1 pixel
     * (all other areas are already covered by masks) has only half the probability to be hit by a random generator
     * as a non-masked area of 2 pixels. Furthermore, a non-masked area of 3 pixels has a higher probability compared
     * to 2 pixels.
     */
    for(int i = 1; i <= csurrArea; i++){
        avgMaskingArea += (double)(i * i);
        tmp += (double)i;
    }
    avgMaskingArea /= tmp;
}

//Distorts the intrinsics for every new stereo configuration
void genStereoSequ::calcDistortedIntrinsics(){
    if(nearZero(pars.distortCamMat.second)){
        for(size_t i = 0; i < R.size(); ++i){
            K1_distorted.emplace_back(K1.clone());
            K2_distorted.emplace_back(K2.clone());
        }
        return;
    }
    for(size_t i = 0; i < R.size(); ++i){
        Mat Kdist1 = K1.clone();
        calcDisortedK(Kdist1);
        K1_distorted.emplace_back(Kdist1.clone());
        Mat Kdist2 = K2.clone();
        calcDisortedK(Kdist2);
        K2_distorted.emplace_back(Kdist2.clone());
    }
}

void genStereoSequ::calcDisortedK(cv::Mat &Kd){
    double f_distort = getRandDoubleValRng(pars.distortCamMat.first, pars.distortCamMat.second)
                       * pow(-1.0, (double)(rand2() % 2));
    double c_distort = getRandDoubleValRng(pars.distortCamMat.first, pars.distortCamMat.second);
    double c__dir_distort = getRandDoubleValRng(0, 2.0 * M_PI);
    f_distort *= Kd.at<double>(0,0);
    f_distort += Kd.at<double>(0,0);
    Kd.at<double>(0,0) = f_distort;
    Kd.at<double>(1,1) = f_distort;

    double cx = Kd.at<double>(0,2);
    double cy = Kd.at<double>(1,2);
    c_distort *= sqrt(cx * cx + cy * cy);
    cx += c_distort * cos(c__dir_distort);
    cy += c_distort * sin(c__dir_distort);
    Kd.at<double>(0,2) = cx;
    Kd.at<double>(1,2) = cy;
}

//Function can only be called after the medium depth for the actual frame is calculated
void genStereoSequ::getImgIntersection(std::vector<cv::Point> &img1Poly,
                                       const cv::Mat &R_use,
                                       const cv::Mat &t_use,
                                       const double depth_use,
                                       bool visualize) {
    CV_Assert(depth_use > 0);
    //Project the corners of img1 on a plane at medium depth in 3D
    vector<Point2f> imgCorners1(4);
    double negXY[2] = {0, 0};
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            Point3d pt((double) (x * (imgSize.width - 1)), (double) (y * (imgSize.height - 1)), 1.0);
            Mat ptm = Mat(pt, false).reshape(1, 3);
            ptm = K1i * ptm;
            ptm *= depth_use / ptm.at<double>(2); //3D position at medium depth
            ptm /= ptm.at<double>(2); //Projection into a plane at medium depth
            imgCorners1[y * 2 + x] = Point2f((float) pt.x, (float) pt.y);
            if (negXY[0] > pt.x) {
                negXY[0] = pt.x;
            }
            if (negXY[1] > pt.y) {
                negXY[1] = pt.y;
            }
        }
    }

    //Project the corners of img2 on a plane at medium depth in 3D
    vector<Point2f> imgCorners2(4);
    Mat A = R_use.t() * K2i;
    Mat a3 = A.row(2);//Get the 3rd row
    double b3 = R_use.col(2).dot(t_use); // R(Col3) * t
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            Point3d pt((double) (x * (imgSize.width - 1)), (double) (y * (imgSize.height - 1)), 1.0);
            Mat ptm = Mat(pt, false).reshape(1, 3);

            //Get scale factor for image coordinate in image 2 to get 3D coordinates at medium depth in the coordinate system of camera 1
            /* From s * x2 = K2 * (R * X + t) with x2...img coordinate in second stereo img, s...scale factor, and X...3D coordinate in coordinate system of camera 1
             * ->we know x2 = (x2_1, x2_2, 1)^T and the distance X_3 of X = (X_1, X_2, X_3)^T
             * Leads to s * R^T * K2^-1 * x2 - R^T * t = X
             * Substituting A = R^T * K2^-1 and b = R^T * t leads to s * A * x2 = X + b
             * As we only need row 3 of the equation: a3 = A_(3,:)...row 3 of A and b3 = b_3...last (3rd) entry of vector b with b3 = R_(:,3) * t
             * Thus we have s * (a3 * x2) = X_3 + b3 which leads to s = (X_3 + b3) / (a3 * x2)
             */
            double a3x2 = a3.dot(ptm.t());//Dot product
            double s = (depth_use + b3) / a3x2;

            ptm = R_use.t() * K2i * ptm;
            ptm *= s; //Scale it to get the correct distance
            ptm -= R_use.t() * t_use;
            //Check, if we have the correct distance
            /*if(!nearZero(ptm.at<double>(2) - depth_use)){
                cout << "Check formulas" << endl;
            }*/
            ptm /= ptm.at<double>(2); //Projection into a plane at medium depth
            imgCorners2[y * 2 + x] = Point2f((float) pt.x, (float) pt.y);
            CV_Assert(nearZero(pt.x - ptm.at<double>(0)) && nearZero(pt.y - ptm.at<double>(1)));
            if (negXY[0] > pt.x) {
                negXY[0] = pt.x;
            }
            if (negXY[1] > pt.y) {
                negXY[1] = pt.y;
            }
        }
    }

    //Transform into positive domain
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            imgCorners1[y * 2 + x].x -= (float) negXY[0];
            imgCorners1[y * 2 + x].y -= (float) negXY[1];
            imgCorners2[y * 2 + x].x -= (float) negXY[0];
            imgCorners2[y * 2 + x].y -= (float) negXY[1];
        }
    }

    //Get the correct order of the points
    vector<Point2f> imgCorners1ch, imgCorners2ch;
    convexHull(imgCorners1, imgCorners1ch);
    convexHull(imgCorners2, imgCorners2ch);
    CV_Assert(imgCorners2ch.size() == 4);

    //Convert into double
    vector<Point2d> imgCorners1chd(4), imgCorners2chd(4);
    for (int k = 0; k < 4; ++k) {
        imgCorners1chd[k] = Point2d((double)imgCorners1ch[k].x, (double)imgCorners1ch[k].y);
        imgCorners2chd[k] = Point2d((double)imgCorners2ch[k].x, (double)imgCorners2ch[k].y);
    }

    //Get the polygon intersection
    polygon_contour c1={0, nullptr, nullptr}, c2={0, nullptr, nullptr}, res={0, nullptr, nullptr};
    add_contour_from_array((double*)&imgCorners1chd[0].x, (int)imgCorners1chd.size(), &c1);
    add_contour_from_array((double*)&imgCorners2chd[0].x, (int)imgCorners2chd.size(), &c2);
    calc_polygon_clipping(POLYGON_OP_INTERSECT, &c1, &c2, &res);
    CV_Assert(res.num_contours == 1);
    std::vector<cv::Point2d> midZPoly;
    for(int v = 0; v < res.contour[0].num_vertices; ++v) {
        midZPoly.emplace_back(cv::Point2d(res.contour[0].vertex[v].x,res.contour[0].vertex[v].y));
    }
    /*for(int c = 0; c < res.num_contours; ++c) {
        std::vector<cv::Point2f> pntsRes;
        for(int v = 0; v < res.contour[c].num_vertices; ++v) {
            pntsRes.push_back(cv::Point2d(res.contour[c].vertex[v].x,res.contour[c].vertex[v].y));
        }
    }*/
    free_polygon(&c1);
    free_polygon(&c2);
    free_polygon(&res);

    //Get rotated rectangles
    /*cv::RotatedRect r1, r2, r3;
    r1 = cv::minAreaRect(imgCorners1);
    r2 = cv::minAreaRect(imgCorners2);


    //Get intersections
    std::vector<cv::Point2f> midZPoly;
    int ti = cv::rotatedRectangleIntersection(r1, r2, midZPoly);

    if (ti == INTERSECT_NONE) {
        throw SequenceException("No intersection of stereo images at medium depth!");
    }*/

    //Transform the points back to their initial coordinate system
    for (auto& i : midZPoly) {
        i.x += negXY[0];
        i.y += negXY[1];
    }

    //Backproject the found intersections to the first image
    std::vector<cv::Point> img1Poly1;
    for (auto& i : midZPoly) {
        Mat ptm = (Mat_<double>(3, 1) << i.x, i.y, 1.0);
        ptm *= depth_use;
        ptm = K1 * ptm;
        ptm /= ptm.at<double>(2);
        img1Poly1.emplace_back(Point((int) round(ptm.at<double>(0)), (int) round(ptm.at<double>(1))));
        if (img1Poly1.back().x > (imgSize.width - 1)) {
            img1Poly1.back().x = imgSize.width - 1;
        }
        if (img1Poly1.back().x < 0) {
            img1Poly1.back().x = 0;
        }
        if (img1Poly1.back().y > (imgSize.height - 1)) {
            img1Poly1.back().y = imgSize.height - 1;
        }
        if (img1Poly1.back().y < 0) {
            img1Poly1.back().y = 0;
        }
    }

    //Get the correct order of the intersections
    vector<int> intSecIdx;
    convexHull(img1Poly1, intSecIdx);
    img1Poly.resize(intSecIdx.size());
    for (size_t j = 0; j < intSecIdx.size(); ++j) {
        img1Poly[j] = img1Poly1[intSecIdx[j]];
    }


    //Draw intersection area
    if (visualize && (verbose & SHOW_STEREO_INTERSECTION)) {
        vector<vector<Point>> intersectCont(1);

        intersectCont[0] = img1Poly;
        Mat wimg = Mat::zeros(imgSize, CV_8UC3);
        drawContours(wimg, intersectCont, 0, Scalar(0, 255, 0), FILLED);
        if(!writeIntermediateImg(wimg, "stereo_intersection")) {
            namedWindow("Stereo intersection", WINDOW_AUTOSIZE);
            imshow("Stereo intersection", wimg);

            waitKey(0);
            destroyWindow("Stereo intersection");
        }
    }
}

//Writes an image of an intermediate result to disk
bool genStereoSequ::writeIntermediateImg(const cv::Mat &img, const std::string &filename){
    if(writeIntermRes_path.empty()){
        return false;
    }
    if(!checkPathExists(writeIntermRes_path)){
        return false;
    }
    if(img.empty()){
        return false;
    }
    Mat tmp;
    if(!((img.type() == CV_8UC1) || (img.type() == CV_8UC3))){
//        cout << "Data for storing into image is not 8bit" << endl;
        int channels = img.channels();
        if(img.type() == CV_16UC1){
            img.convertTo(tmp, CV_8UC1, 1./256.);
        }else if(img.type() == CV_16UC3){
            img.convertTo(tmp, CV_8UC3, 1./256.);
        }else if ((img.type() == CV_64F) || (img.type() == CV_32F)){
            double minv, maxv;
            if(channels == 1) {
                cv::minMaxLoc(img, &minv, &maxv);
                img.copyTo(tmp);
                tmp.convertTo(tmp, CV_64FC1);
                tmp -= minv;
                tmp.convertTo(tmp, CV_8UC1, 255./(maxv - minv));
            }else if (channels == 3){
                vector<Mat> channels;
                cv::split(img, channels);
                for(auto &ch: channels){
                    cv::minMaxLoc(ch, &minv, &maxv);
                    ch.convertTo(tmp, CV_64FC1);
                    ch -= minv;
                    double scale = 255./(maxv - minv);
                    ch.convertTo(ch, CV_8UC1, scale, scale * minv);
                }
                cv::merge(channels, tmp);
            }else{
                return false;
            }
        }else{
            double minv, maxv;
            if(channels == 1){
                cv::minMaxLoc(img, &minv, &maxv);
                img.copyTo(tmp);
                tmp.convertTo(tmp, CV_64FC1);
                tmp -= minv;
                tmp.convertTo(tmp, CV_8UC1, 255./(maxv - minv));
            }else if (channels == 3){
                vector<Mat> channels;
                cv::split(img, channels);
                for(auto &ch: channels){
                    cv::minMaxLoc(ch, &minv, &maxv);
                    ch.convertTo(tmp, CV_64FC1);
                    ch -= minv;
                    double scale = 255./(maxv - minv);
                    ch.convertTo(ch, CV_8UC1, scale, scale * minv);
                }
                cv::merge(channels, tmp);
            }else{
                return false;
            }
        }
    }else{
        tmp = img;
    }
    string filen = filename + "_frame_" + std::to_string(actFrameCnt);
    string resPath = concatPath(writeIntermRes_path, filen + ".png");
    int cnt = 0;
    while(checkFileExists(resPath)){
        resPath = concatPath(writeIntermRes_path, filen + "_" + std::to_string(cnt) + ".png");
        cnt++;
    }
    cv::imwrite(resPath, tmp);
    return true;
}

//Get the fraction of intersection between the stereo images for every image region (3x3)
void genStereoSequ::getInterSecFracRegions(cv::Mat &fracUseableTPperRegion_,
                                           const cv::Mat &R_use,
                                           const cv::Mat &t_use,
                                           const double depth_use,
                                           cv::InputArray mask,
                                           cv::OutputArray imgUsableMask) {
    //Check overlap of the stereo images
    std::vector<cv::Point> img1Poly;
    getImgIntersection(img1Poly, R_use, t_use, depth_use, mask.empty());

    //Create a mask for overlapping views
    vector<vector<Point>> intersectCont(1);

    intersectCont[0] = img1Poly;
    Mat wimg = Mat::zeros(imgSize, CV_8UC1);
    drawContours(wimg, intersectCont, 0, Scalar(255), FILLED);

    if (!mask.empty()) {
        wimg &= mask.getMat();
    }

    if (imgUsableMask.needed()) {
        if (imgUsableMask.empty()) {
            imgUsableMask.create(imgSize, CV_8UC1);
        }
        Mat outmat_tmp = imgUsableMask.getMat();
        wimg.copyTo(outmat_tmp);
    }

    fracUseableTPperRegion_ = Mat(3, 3, CV_64FC1);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            if (mask.empty()) {
                fracUseableTPperRegion_.at<double>(y, x) =
                        (double) countNonZero(wimg(regROIs[y][x])) / (double) regROIs[y][x].area();
            } else {
                Mat regMask = mask.getMat()(regROIs[y][x]);
                int cntNZMat = countNonZero(regMask);
                if(cntNZMat == 0){
                    fracUseableTPperRegion_.at<double>(y, x) = 0;
                }
                else {
                    fracUseableTPperRegion_.at<double>(y, x) =
                            (double) countNonZero(wimg(regROIs[y][x])) /
                            (double) cntNZMat;
                }
            }
        }
    }

}

//Get the fraction of intersection between the stereo images for every image region (3x3) and for various depths (combined, smallest area)
void genStereoSequ::getInterSecFracRegions(cv::Mat &fracUseableTPperRegion_,
                            const cv::Mat &R_use,
                            const cv::Mat &t_use,
                            const std::vector<double> &depth_use,
                            cv::InputArray mask,
                            cv::OutputArray imgUsableMask){
    //Check overlap of the stereo images
    Mat combDepths = Mat::ones(imgSize, CV_8UC1) * 255;
    for(auto &d: depth_use) {
        std::vector<cv::Point> img1Poly;
        getImgIntersection(img1Poly, R_use, t_use, d, false);

        //Create a mask for overlapping views
        vector<vector<Point>> intersectCont(1);

        intersectCont[0] = img1Poly;
        Mat wimg = Mat::zeros(imgSize, CV_8UC1);
        drawContours(wimg, intersectCont, 0, Scalar(255), FILLED);

        combDepths &= wimg;
    }

    //Draw intersection area
    if(mask.empty() && (verbose & SHOW_STEREO_INTERSECTION)){
        if(!writeIntermediateImg(combDepths, "stereo_intersection_combDepths")) {
            namedWindow("Stereo intersection", WINDOW_AUTOSIZE);
            imshow("Stereo intersection", combDepths);
            waitKey(0);
            destroyWindow("Stereo intersection");
        }
    }

    if (!mask.empty()) {
        combDepths &= mask.getMat();
    }

    if (imgUsableMask.needed()) {
        if (imgUsableMask.empty()) {
            imgUsableMask.create(imgSize, CV_8UC1);
        }
        Mat outmat_tmp = imgUsableMask.getMat();
        combDepths.copyTo(outmat_tmp);
    }

    fracUseableTPperRegion_ = Mat(3, 3, CV_64FC1);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            if (mask.empty()) {
                fracUseableTPperRegion_.at<double>(y, x) =
                        (double) countNonZero(combDepths(regROIs[y][x])) / (double) regROIs[y][x].area();
            } else {
                Mat regMask = mask.getMat()(regROIs[y][x]);
                int cntNZMat = countNonZero(regMask);
                if(cntNZMat == 0){
                    fracUseableTPperRegion_.at<double>(y, x) = 0;
                }
                else {
                    fracUseableTPperRegion_.at<double>(y, x) =
                            (double) countNonZero(combDepths(regROIs[y][x])) /
                            (double) cntNZMat;
                }
            }
        }
    }
}

//Get number of correspondences per image and Correspondences per image regions
//Check if there are too many correspondences per region as every correspondence needs a minimum distance to its neighbor. If yes, the minimum distance and/or number of correspondences are adapted.
void genStereoSequ::initNrCorrespondences() {
    //Number of correspondences per image
    genNrCorrsImg();

    //Correspondences per image regions
    bool res = initFracCorrImgReg();
    while (!res) {
        genNrCorrsImg();
        res = initFracCorrImgReg();
    }
}

//Initialize fraction of correspondences per image region and calculate the absolute number of TP/TN correspondences per image region
bool genStereoSequ::initFracCorrImgReg() {
    if ((pars.corrsPerRegRepRate == 0) && pars.corrsPerRegion.empty()) {
        for (size_t i = 0; i < totalNrFrames; i++) {
            Mat newCorrsPerRegion(3, 3, CV_64FC1);
            double sumNewCorrsPerRegion = 0;
            while(nearZero(sumNewCorrsPerRegion)) {
                cv::randu(newCorrsPerRegion, Scalar(0), Scalar(1.0));
                sumNewCorrsPerRegion = sum(newCorrsPerRegion)[0];
            }
            newCorrsPerRegion /= sumNewCorrsPerRegion;
            pars.corrsPerRegion.push_back(newCorrsPerRegion.clone());
        }
        pars.corrsPerRegRepRate = 1;
    } else if (pars.corrsPerRegRepRate == 0) {
        pars.corrsPerRegRepRate = std::max<size_t>(totalNrFrames / pars.corrsPerRegion.size(), 1);
    } else if (pars.corrsPerRegion.empty()) {
        //Randomly initialize the fractions
        size_t nrMats = std::max<size_t>(totalNrFrames / pars.corrsPerRegRepRate, 1);
        for (size_t i = 0; i < nrMats; i++) {
            Mat newCorrsPerRegion(3, 3, CV_64FC1);
            double sumNewCorrsPerRegion = 0;
            while(nearZero(sumNewCorrsPerRegion)) {
                cv::randu(newCorrsPerRegion, Scalar(0), Scalar(1.0));
                sumNewCorrsPerRegion = sum(newCorrsPerRegion)[0];
            }
            newCorrsPerRegion /= sumNewCorrsPerRegion;
            pars.corrsPerRegion.push_back(newCorrsPerRegion.clone());
        }
    }

    for (auto&& k : pars.corrsPerRegion) {
        double regSum = sum(k)[0];
        if (!nearZero(regSum) && !nearZero(regSum - 1.0))
            k /= regSum;
        else if (nearZero(regSum)) {
            k = Mat::ones(3, 3, CV_64FC1) / 9.0;
        }
    }

    //Generate absolute number of correspondences per image region and frame
    nrTruePosRegs.clear();
    nrCorrsRegs.clear();
    nrTrueNegRegs.clear();
    nrTruePosRegs.reserve(totalNrFrames);
    nrCorrsRegs.reserve(totalNrFrames);
    nrTrueNegRegs.reserve(totalNrFrames);
    size_t cnt = 0, stereoIdx = 0;
    cv::Mat fracUseableTPperRegion_tmp, fracUseableTPperRegionNeg, fUTPpRNSum1, stereoImgsOverlapMask_tmp;
    for (size_t i = 0; i < totalNrFrames; i++) {
        //Get intersection of stereo images at medium distance plane for every image region in camera 1
        if ((i % pars.nFramesPerCamConf) == 0) {
            if (i > 0) {
                stereoIdx++;
            }

            vector<double> combDepths(3);
            combDepths[0] = depthMid[stereoIdx];
            combDepths[1] = depthFar[stereoIdx] + (maxFarDistMultiplier - 1.0) * depthFar[stereoIdx] * pars.corrsPerDepth.far;
            combDepths[2] = depthNear[stereoIdx] + max((1.0 - pars.corrsPerDepth.near), 0.2) * (depthMid[stereoIdx] - depthNear[stereoIdx]);

            getInterSecFracRegions(fracUseableTPperRegion_tmp,
                                   R[stereoIdx],
                                   t[stereoIdx],
                                   combDepths,
                                   cv::noArray(),
                                   stereoImgsOverlapMask_tmp);
            stereoImgsOverlapMask.push_back(stereoImgsOverlapMask_tmp.clone());
            fracUseableTPperRegion.push_back(fracUseableTPperRegion_tmp.clone());
            fracUseableTPperRegionNeg = Mat::ones(3, 3, CV_64FC1) - fracUseableTPperRegion_tmp;
            double sumFracUseableTPperRegionNeg = cv::sum(fracUseableTPperRegionNeg)[0];
            if(nearZero(sumFracUseableTPperRegionNeg))
                sumFracUseableTPperRegionNeg = 1.0;
            fUTPpRNSum1 = fracUseableTPperRegionNeg / sumFracUseableTPperRegionNeg;
        }

        cv::Mat corrsTooMuch;
        double activeRegions = 9.0;
        Mat corrsRemain = Mat::ones(3, 3, CV_32SC1);

        //Get number of correspondences per region
        Mat newCorrsPerRegion;
        Mat negsReg(3, 3, CV_64FC1);
        bool recalcCorrsPerRegion = false;
        do {
            newCorrsPerRegion = pars.corrsPerRegion[cnt] * (double)nrCorrs[i];
            newCorrsPerRegion.convertTo(newCorrsPerRegion, CV_32SC1, 1.0, 0.5);//Corresponds to round
            int32_t chkSize = (int32_t)sum(newCorrsPerRegion)[0] - (int32_t) nrCorrs[i];
            if (chkSize > 0) {
                do {
                    int pos = (int)(rand2() % 9);
                    if (newCorrsPerRegion.at<int32_t>(pos) > 0) {
                        newCorrsPerRegion.at<int32_t>(pos)--;
                        chkSize--;
                    } /*else
                {
				    cout << "Zero corrs in region " << pos << "of frame " << i << endl;
                }*/
                } while (chkSize > 0);
            } else if (chkSize < 0) {
                do {
                    int pos = (int)(rand2() % 9);
                    if (!nearZero(pars.corrsPerRegion[cnt].at<double>(pos))) {
                        newCorrsPerRegion.at<int32_t>(pos)++;
                        chkSize++;
                    }
                } while (chkSize < 0);
            }

            //Check, if the overall number of correspondences is still correct
            if(verbose & PRINT_WARNING_MESSAGES) {
                int corrsDiff = (int)sum(newCorrsPerRegion)[0] - (int)nrCorrs[i];
                if (((double) abs(corrsDiff) / (double) nrCorrs[i]) > 0.01) {
                    cout << "The distribution of correspondences over the regions did not work!" << endl;
                }
            }

            Mat usedTNonNoOverlapRat;
            Mat newCorrsPerRegiond;
            int cnt1 = 0;
            const int cnt1Max = 3;
            while ((cv::sum(corrsRemain)[0] > 0) && (cnt1 < cnt1Max)) {
                //Check if there are too many correspondences per region as every correspondence needs a minimum distance to its neighbor
                double minCorr, maxCorr;
                cv::minMaxLoc(newCorrsPerRegion, &minCorr, &maxCorr);
                if(nearZero(activeRegions)) activeRegions = 0.1;
                double regA = activeRegions * (double) imgSize.area() / (9.0 * 9.0);
                double areaCorrs = maxCorr * avgMaskingArea *
                                   enlargeKPDist;//Multiply by 1.15 to take gaps into account that are a result of randomness

                if (areaCorrs > regA) {
                    if(verbose & PRINT_WARNING_MESSAGES) {
                        cout << "There are too many keypoints per region when demanding a minimum keypoint distance of "
                             << pars.minKeypDist << ". Changing it!" << endl;
                    }
                    double mKPdist = floor((sqrt(regA / (enlargeKPDist * maxCorr)) - 1.0) / 2.0) - DBL_EPSILON;
                    if (mKPdist <= 1.414214) {
                        pars.minKeypDist = 1.0 - DBL_EPSILON;
                        genMasks();
                        //Get max # of correspondences
                        double maxFC = (double) *std::max_element(nrCorrs.begin(), nrCorrs.end());
                        //Get the largest portion of correspondences within a single region
                        vector<double> cMaxV(pars.corrsPerRegion.size());
                        for (size_t k = 0; k < pars.corrsPerRegion.size(); k++) {
                            cv::minMaxLoc(pars.corrsPerRegion[k], &minCorr, &maxCorr);
                            cMaxV[k] = maxCorr;
                        }
                        maxCorr = *std::max_element(cMaxV.begin(), cMaxV.end());
                        maxCorr *= maxFC;
                        //# KPs reduction factor
                        double reduF = regA / (enlargeKPDist * avgMaskingArea * maxCorr);
                        if(reduF < 1.0) {
                            if (verbose & PRINT_WARNING_MESSAGES) {
                                cout
                                        << "Changed the minimum keypoint distance to 1.0. There are still too many keypoints. Changing the number of keypoints!"
                                        << endl;
                            }
                            //Get worst inlier ratio
                            double minILR = *std::min_element(inlRat.begin(), inlRat.end());
                            //Calc max true positives
                            auto maxTPNew = (size_t) floor(maxCorr * reduF * minILR);
                            if (verbose & PRINT_WARNING_MESSAGES) {
                                cout << "Changing max. true positives to " << maxTPNew << endl;
                            }
                            if ((pars.truePosRange.second - pars.truePosRange.first) == 0) {
                                pars.truePosRange.first = pars.truePosRange.second = maxTPNew;
                            } else {
                                if (pars.truePosRange.first >= maxTPNew) {
                                    pars.truePosRange.first = maxTPNew / 2;
                                    pars.truePosRange.second = maxTPNew;
                                } else {
                                    pars.truePosRange.second = maxTPNew;
                                }
                            }
                            nrTruePosRegs.clear();
                            nrCorrsRegs.clear();
                            nrTrueNegRegs.clear();
                            return false;
                        }else{
                            if(verbose & PRINT_WARNING_MESSAGES) {
                                cout << "Changed the minimum keypoint distance to " << pars.minKeypDist << endl;
                            }
                        }
                    } else {
                        if(verbose & PRINT_WARNING_MESSAGES) {
                            cout << "Changed the minimum keypoint distance to " << mKPdist << endl;
                        }
                        pars.minKeypDist = mKPdist;
                        genMasks();
                    }
                }

                //Get number of true negatives per region

                if (cnt1 == 0) {
                    int maxit = 5;
                    maxCorr = 1.0;
                    while ((maxCorr > 0.5) && (maxit > 0)) {
                        negsReg = Mat(3, 3, CV_64FC1);
                        double sumNegsReg = 0;
                        while(nearZero(sumNegsReg)) {
                            cv::randu(negsReg, Scalar(0), Scalar(1.0));
                            sumNegsReg = sum(negsReg)[0];
                        }
                        negsReg /= sumNegsReg;
                        negsReg = 0.33 * negsReg + negsReg.mul(fUTPpRNSum1);
                        negsReg /= sum(negsReg)[0];
                        cv::minMaxLoc(negsReg, &minCorr, &maxCorr);
                        maxit--;
                    }
                    if (maxit == 0) {
                        minCorr = maxCorr / 16.0;
                        for (int y = 0; y < 3; ++y) {
                            for (int x = 0; x < 3; ++x) {
                                if (nearZero(negsReg.at<double>(y, x) - maxCorr)) {
                                    negsReg.at<double>(y, x) /= 2.0;
                                } else {
                                    negsReg.at<double>(y, x) += minCorr;
                                }
                            }
                        }
                        negsReg /= sum(negsReg)[0];
                    }
                    newCorrsPerRegion.convertTo(newCorrsPerRegiond, CV_64FC1);
                    corrsTooMuch = fracUseableTPperRegionNeg.mul(newCorrsPerRegiond);
                    corrsTooMuch.convertTo(corrsTooMuch, CV_32SC1, 1.0, 0.5);//Corresponds to round
                } else {
                    negsReg.convertTo(negsReg, CV_64FC1);
                    negsReg = 0.66 * negsReg + 0.33 * negsReg.mul(fUTPpRNSum1);
                    double sumNegsReg = sum(negsReg)[0];
                    if(!nearZero(sumNegsReg))
                        negsReg /= sumNegsReg;
                    cv::minMaxLoc(negsReg, &minCorr, &maxCorr);
                    if (maxCorr > 0.5) {
                        minCorr = maxCorr / 16.0;
                        for (int y = 0; y < 3; ++y) {
                            for (int x = 0; x < 3; ++x) {
                                if (nearZero(negsReg.at<double>(y, x) - maxCorr)) {
                                    negsReg.at<double>(y, x) /= 2.0;
                                } else {
                                    negsReg.at<double>(y, x) += minCorr;
                                }
                            }
                        }
                        negsReg /= sum(negsReg)[0];
                    }
                }
                negsReg = negsReg.mul(newCorrsPerRegiond);//Max number of true negatives per region
                double sumNegsReg = sum(negsReg)[0];
                if(!nearZero(sumNegsReg))
                    negsReg *= (double) nrTrueNeg[i] / sumNegsReg;
                negsReg.convertTo(negsReg, CV_32SC1, 1.0, 0.5);//Corresponds to round
                for (size_t j = 0; j < 9; j++) {
                    while (negsReg.at<int32_t>(j) > newCorrsPerRegion.at<int32_t>(j))
                        negsReg.at<int32_t>(j)--;
                }
                chkSize = (int32_t) sum(negsReg)[0] - (int32_t) nrTrueNeg[i];
                if (chkSize > 0) {
                    do {
                        int pos = (int)(rand2() % 9);
                        if (negsReg.at<int32_t>(pos) > 0) {
                            negsReg.at<int32_t>(pos)--;
                            chkSize--;
                        } /*else
                {
                    cout << "Zero neg corrs in region " << pos << "of frame " << i << endl;
                }*/
                    } while (chkSize > 0);
                } else if (chkSize < 0) {
                    do {
                        int pos = (int)(rand2() % 9);
                        if (negsReg.at<int32_t>(pos) < newCorrsPerRegion.at<int32_t>(pos)) {
                            negsReg.at<int32_t>(pos)++;
                            chkSize++;
                        }
                    } while (chkSize < 0);
                }

                //Check, if there are still TP outside the intersection of the 2 stereo camera views
                corrsRemain = corrsTooMuch - negsReg;
                usedTNonNoOverlapRat = Mat::ones(3, 3, CV_64FC1);
                double areaPerKP = avgMaskingArea * enlargeKPDist;
                for (int y = 0; y < 3; ++y) {
                    for (int x = 0; x < 3; ++x) {
                        if (corrsRemain.at<int32_t>(y, x) > 0) {
                            usedTNonNoOverlapRat.at<double>(y, x) =
                                    (double) negsReg.at<int32_t>(y, x) / (double) corrsTooMuch.at<int32_t>(y, x);
                            double neededArea =
                                    (newCorrsPerRegiond.at<double>(y, x) - (double) negsReg.at<int32_t>(y, x)) *
                                    areaPerKP;
                            double usableArea =
                                    fracUseableTPperRegion_tmp.at<double>(y, x) * (double) regROIs[y][x].area();
                            if (neededArea > usableArea) {
                                double diffArea = neededArea - usableArea;
                                corrsRemain.at<int32_t>(y, x) = (int32_t) round(diffArea / areaPerKP);
                            } else {
                                corrsRemain.at<int32_t>(y, x) = 0;
                            }

                        } else {
                            corrsRemain.at<int32_t>(y, x) = 0;
                        }
                    }
                }
                activeRegions = cv::sum(usedTNonNoOverlapRat)[0];

                cnt1++;
            }
            if ((cv::sum(corrsRemain)[0] > 0) && (cnt1 >= cnt1Max)) {
                //Adapt number of correspondences per region
                pars.corrsPerRegion[cnt] = pars.corrsPerRegion[cnt].mul(usedTNonNoOverlapRat);
                double sumCorrsPerRegion = cv::sum(pars.corrsPerRegion[cnt])[0];
                if(nearZero(sumCorrsPerRegion)){
                    double sumNewCorrsPerRegion = 0;
                    while(nearZero(sumNewCorrsPerRegion)) {
                        cv::randu(pars.corrsPerRegion[cnt], Scalar(0), Scalar(1.0));
                        sumNewCorrsPerRegion = sum(pars.corrsPerRegion[cnt])[0];
                    }
                    newCorrsPerRegion /= sumNewCorrsPerRegion;
                    sumCorrsPerRegion = 1.0;
                }
                pars.corrsPerRegion[cnt] /= sumCorrsPerRegion;
                recalcCorrsPerRegion = true;
                cnt1 = 0;
                corrsRemain = Mat::ones(3, 3, CV_32SC1);
                activeRegions = 9.0;
            } else {
                recalcCorrsPerRegion = false;
            }
        } while (recalcCorrsPerRegion);

        nrCorrsRegs.push_back(newCorrsPerRegion.clone());
        nrTrueNegRegs.push_back(negsReg.clone());

        //Check, if the overall number of correspondences is still correct
        if(verbose & PRINT_WARNING_MESSAGES) {
            int corrsDiff = (int)sum(newCorrsPerRegion)[0] - (int)nrCorrs[i];
            if (((double) abs(corrsDiff) / (double) nrCorrs[i]) > 0.01) {
                cout << "The distribution of correspondences over the regions did not work!" << endl;
            }
        }

        //Get number of true positives per region
        newCorrsPerRegion = newCorrsPerRegion - negsReg;
        nrTruePosRegs.push_back(newCorrsPerRegion.clone());

        //Check the overall inlier ratio
        if(verbose & PRINT_WARNING_MESSAGES) {
            int32_t allRegTP = (int32_t)sum(nrTruePosRegs.back())[0];
            int32_t allRegCorrs = (int32_t)sum(nrCorrsRegs.back())[0];
            double thisInlRatDiff = (double)allRegTP / (double)allRegCorrs - inlRat[i];
            double testVal = min((double)allRegCorrs / 100.0, 1.0) * thisInlRatDiff / 100.0;
            if (!nearZero(testVal)) {
                cout << "Initial inlier ratio differs from given values (0 - 1.0) by "
                     << thisInlRatDiff << endl;
            }
        }

        //Check if the fraction of corrspondences per region must be changend
        if ((((i + 1) % (pars.corrsPerRegRepRate)) == 0)) {
            cnt++;
            if (cnt >= pars.corrsPerRegion.size()) {
                cnt = 0;
            }
        }
    }

    return true;
}

//Generate number of correspondences
void genStereoSequ::genNrCorrsImg() {
    nrCorrs.resize(totalNrFrames);
    nrTrueNeg.resize(totalNrFrames);
    if ((pars.truePosRange.second - pars.truePosRange.first) == 0) {
        if (pars.truePosRange.first == 0) {
            throw SequenceException(
                    "Number of true positives specified 0 for all frames - nothing can be generated!\n");
        }
        nrTruePos.resize(totalNrFrames, pars.truePosRange.first);
        for (size_t i = 0; i < totalNrFrames; i++) {
            nrCorrs[i] = (size_t) round((double) pars.truePosRange.first / inlRat[i]);
            nrTrueNeg[i] = nrCorrs[i] - pars.truePosRange.first;
        }
        if (nearZero(pars.inlRatRange.first - pars.inlRatRange.second)) {
            fixedNrCorrs = true;
        }
    } else {
        size_t initTruePos = std::max((size_t) round(
                getRandDoubleValRng((double) pars.truePosRange.first, (double) pars.truePosRange.second)), (size_t) 1);
        if (nearZero(pars.truePosChanges)) {
            nrTruePos.resize(totalNrFrames, initTruePos);
            for (size_t i = 0; i < totalNrFrames; i++) {
                nrCorrs[i] = (size_t) round((double) initTruePos / inlRat[i]);
                nrTrueNeg[i] = nrCorrs[i] - initTruePos;
            }
        } else if (nearZero(pars.truePosChanges - 100.0)) {
            nrTruePos.resize(totalNrFrames);
            std::uniform_int_distribution<size_t> distribution(pars.truePosRange.first, pars.truePosRange.second);
            for (size_t i = 0; i < totalNrFrames; i++) {
                nrTruePos[i] = distribution(rand_gen);
                nrCorrs[i] = (size_t) round((double) nrTruePos[i] / inlRat[i]);
                nrTrueNeg[i] = nrCorrs[i] - nrTruePos[i];
            }
        } else {
            nrTruePos.resize(totalNrFrames);
            nrTruePos[0] = initTruePos;
            nrCorrs[0] = (size_t) round((double) nrTruePos[0] / inlRat[0]);
            nrTrueNeg[0] = nrCorrs[0] - nrTruePos[0];
            for (size_t i = 1; i < totalNrFrames; i++) {
                auto rangeVal = (size_t) round(pars.truePosChanges * (double) nrTruePos[i - 1]);
                size_t maxTruePos = nrTruePos[i - 1] + rangeVal;
                maxTruePos = maxTruePos > pars.truePosRange.second ? pars.truePosRange.second : maxTruePos;
                int64_t minTruePos_ = (int64_t) nrTruePos[i - 1] - (int64_t) rangeVal;
                size_t minTruePos = minTruePos_ < 0 ? 0 : ((size_t) minTruePos_);
                minTruePos = minTruePos < pars.truePosRange.first ? pars.truePosRange.first : minTruePos;
                std::uniform_int_distribution<size_t> distribution(minTruePos, maxTruePos);
                nrTruePos[i] = distribution(rand_gen);
                nrCorrs[i] = (size_t) round((double) nrTruePos[i] / inlRat[i]);
                nrTrueNeg[i] = nrCorrs[i] - nrTruePos[i];
            }
        }
    }
}

//Generate the inlier ratio for every frame
void genStereoSequ::genInlierRatios() {
    if (nearZero(pars.inlRatRange.first - pars.inlRatRange.second)) {
        inlRat.resize(totalNrFrames, max(pars.inlRatRange.first, 0.01));
    } else {
        double initInlRat = getRandDoubleValRng(pars.inlRatRange.first, pars.inlRatRange.second, rand_gen);
        initInlRat = max(initInlRat, 0.01);
        if (nearZero(pars.inlRatChanges)) {
            inlRat.resize(totalNrFrames, initInlRat);
        } else if (nearZero(pars.inlRatChanges - 100.0)) {
            inlRat.resize(totalNrFrames);
            std::uniform_real_distribution<double> distribution(pars.inlRatRange.first, pars.inlRatRange.second);
            for (size_t i = 0; i < totalNrFrames; i++) {
                inlRat[i] = max(distribution(rand_gen), 0.01);
            }
        } else {
            inlRat.resize(totalNrFrames);
            inlRat[0] = initInlRat;
            for (size_t i = 1; i < totalNrFrames; i++) {
                double maxInlrat = inlRat[i - 1] + pars.inlRatChanges * inlRat[i - 1];
                maxInlrat = maxInlrat > pars.inlRatRange.second ? pars.inlRatRange.second : maxInlrat;
                double minInlrat = inlRat[i - 1] - pars.inlRatChanges * inlRat[i - 1];
                minInlrat = minInlrat < pars.inlRatRange.first ? pars.inlRatRange.first : minInlrat;
                inlRat[i] = max(getRandDoubleValRng(minInlrat, maxInlrat), 0.01);
            }
        }
    }
}

/* Constructs an absolute camera path including the position and rotation of the stereo rig (left/lower camera centre)
*/
void genStereoSequ::constructCamPath() {
    //Calculate the absolute velocity of the cameras
    absCamVelocity = 0;
    for (auto& i : t) {
        absCamVelocity += norm(i);
    }
    absCamVelocity /= (double) t.size();
    absCamVelocity *= pars.relCamVelocity;//in baselines from frame to frame

    //Calculate total number of frames
//    totalNrFrames = pars.nFramesPerCamConf * t.size();
    if((pars.nTotalNrFrames >= max(pars.nFramesPerCamConf * (t.size() - 1) + 1, pars.nFramesPerCamConf))
    && (pars.nTotalNrFrames <= pars.nFramesPerCamConf * t.size())) {
        totalNrFrames = pars.nTotalNrFrames;
    }else{
        if(pars.nTotalNrFrames <= pars.nFramesPerCamConf * t.size()) {
            cout << "The provided number of total frames would be too small to use all provided stereo configurations."
                 << endl;
        }else{
            cout << "The provided number of total frames is too large. The sequence of different "
                    "stereo configurations would be repeated." << endl;
        }
        totalNrFrames = pars.nFramesPerCamConf * t.size();
        cout << "Changing number of frames from " << pars.nTotalNrFrames << " to " << totalNrFrames << endl;
    }

    //Number of track elements
    size_t nrTracks = pars.camTrack.size();

    absCamCoordinates = vector<Poses>(totalNrFrames);
    Mat R0;
    if (pars.R.empty())
        R0 = Mat::eye(3, 3, CV_64FC1);
    else
        R0 = pars.R;
    Mat t1 = Mat::zeros(3, 1, CV_64FC1);
    if (nrTracks == 1) {
        double camTrackNorm = norm(pars.camTrack[0]);
        Mat R1;
        if(!nearZero(camTrackNorm)) {
            pars.camTrack[0] /= camTrackNorm;
            R1 = R0 * getTrackRot(pars.camTrack[0]);
        }
        else{
            R1 = R0;
        }
        Mat t_piece = absCamVelocity * pars.camTrack[0];
        absCamCoordinates[0] = Poses(R1.clone(), t1.clone());
        for (size_t i = 1; i < totalNrFrames; i++) {
            t1 += t_piece;
            absCamCoordinates[i] = Poses(R1.clone(), t1.clone());
        }
    } else {
        //Get differential vectors of the path and the overall path length
        vector<Mat> diffTrack = vector<Mat>(nrTracks - 1);
        vector<double> tdiffNorms = vector<double>(nrTracks - 1);
        double trackNormSum = 0;//norm(pars.camTrack[0]);
//		diffTrack[0] = pars.camTrack[0].clone();// / trackNormSum;
//		tdiffNorms[0] = trackNormSum;
        for (size_t i = 0; i < nrTracks - 1; i++) {
            Mat tdiff = pars.camTrack[i + 1] - pars.camTrack[i];
            double tdiffnorm = norm(tdiff);
            trackNormSum += tdiffnorm;
            diffTrack[i] = tdiff.clone();// / tdiffnorm;
            tdiffNorms[i] = tdiffnorm;
        }

        //Calculate a new scaling for the path based on the original path length, total number of frames and camera velocity
        if(nearZero(trackNormSum)){
            throw SequenceException("Provided a track without movement!");
        }
        double trackScale = (double) (totalNrFrames - 1) * absCamVelocity / trackNormSum;
        //Rescale track diffs
        for (size_t i = 0; i < nrTracks - 1; i++) {
            diffTrack[i] *= trackScale;
            tdiffNorms[i] *= trackScale;
        }

        //Get camera positions
        Mat R_track = getTrackRot(diffTrack[0]);
        Mat R_track_old = R_track.clone();
        Mat R1 = R0 * R_track;
        pars.camTrack[0].copyTo(t1);
        t1 *= trackScale;
        absCamCoordinates[0] = Poses(R1.clone(), t1.clone());
        double actDiffLength = 0;
        size_t actTrackNr = 0, lastTrackNr = 0;
        for (size_t i = 1; i < totalNrFrames; i++) {
            bool firstAdd = true;
            Mat multTracks = Mat::zeros(3, 1, CV_64FC1);
            double usedLength = 0;
            while ((actDiffLength < (absCamVelocity - DBL_EPSILON)) && (actTrackNr < (nrTracks - 1))) {
                if (firstAdd) {
                    if(!nearZero(tdiffNorms[lastTrackNr]))
                        multTracks += actDiffLength * diffTrack[lastTrackNr] / tdiffNorms[lastTrackNr];
                    usedLength = actDiffLength;
                    firstAdd = false;
                } else {
                    multTracks += diffTrack[lastTrackNr];
                    usedLength += tdiffNorms[lastTrackNr];
                }

                lastTrackNr = actTrackNr;

                actDiffLength += tdiffNorms[actTrackNr++];
            }
            if(!nearZero(tdiffNorms[lastTrackNr]))
                multTracks += (absCamVelocity - usedLength) * diffTrack[lastTrackNr] / tdiffNorms[lastTrackNr];

            R_track = getTrackRot(diffTrack[lastTrackNr], R_track_old);
            R_track_old = R_track.clone();
            R1 = R0 * R_track;
            t1 += multTracks;
            absCamCoordinates[i] = Poses(R1.clone(), t1.clone());
            actDiffLength -= absCamVelocity;
        }
    }

    if (verbose & SHOW_INIT_CAM_PATH)
        visualizeCamPath();
}

/*Calculates a rotation for every differential vector of a track segment to ensure that the camera looks always in the direction of the track segment.
* If the track segment equals the x-axis, the camera faces into positive x-direction (if the initial rotaion equals the identity).
* The y axis always points down as the up vector is defined as [0,-1,0]. Thus, there is no roll in the camera rotation.
*/
cv::Mat genStereoSequ::getTrackRot(const cv::Mat tdiff, cv::InputArray R_old) {
    CV_Assert((tdiff.rows == 3) && (tdiff.cols == 1) && (tdiff.type() == CV_64FC1));

    Mat R_C2W = Mat::eye(3, 3, CV_64FC1);

    if (nearZero(cv::norm(tdiff)))
        return R_C2W;

    Mat tdiff_ = tdiff.clone();
    tdiff_ /= norm(tdiff_);

    Mat Rold;
    if (!R_old.empty())
        Rold = R_old.getMat();


    //Define up-vector as global -y axis
    Mat world_up = (Mat_<double>(3, 1) << 0, -1, 0);
//    Mat world_up = (Mat_<double>(3, 1) << 0, 1, 0);
    world_up /= norm(world_up);

    if (nearZero(cv::sum(abs(tdiff_ - world_up))[0])) {
        R_C2W = (Mat_<double>(3, 3) << 1.0, 0, 0,
                0, 0, -1.0,
                0, 1.0, 0);
        if (!Rold.empty()) {
            Mat Rr;
            if (roundR(Rold, Rr, R_C2W)) {
                R_C2W = Rr;
            }
        }
    } else if (nearZero(cv::sum(abs(tdiff_ + world_up))[0])) {
        R_C2W = (Mat_<double>(3, 3) << 1.0, 0, 0,
                0, 0, 1.0,
                0, -1.0, 0);
        if (!Rold.empty()) {
            Mat Rr;
            if (roundR(Rold, Rr, R_C2W)) {
                R_C2W = Rr;
            }
        }
    } else {
        //Get local axis that is perpendicular to up-vector and look-at vector (new x axis) -> to prevent roll
        Mat xa = tdiff_.cross(world_up);
        xa /= norm(xa);
        //Get local axis that is perpendicular to new x-axis and look-at vector (new y axis)
        Mat ya = tdiff_.cross(xa);
        ya /= norm(ya);

        //Build the rotation matrix (camera to world: x_world = R_C2W * x_local + t) by stacking the normalized axis vectors as columns of R=[x,y,z]
        xa.copyTo(R_C2W.col(0));
        ya.copyTo(R_C2W.col(1));
        tdiff_.copyTo(R_C2W.col(2));
    }

    return R_C2W;//return rotation from camera to world
}

void genStereoSequ::visualizeCamPath() {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Camera path"));
    initPCLViewerCoordinateSystems(viewer);

    for (auto &i : absCamCoordinates) {
        addVisualizeCamCenter(viewer, i.R, i.t);
    }

    viewer->initCameraParameters();

    startPCLViewer(viewer);
}

//Calculate the thresholds for the depths near, mid, and far for every camera configuration
bool genStereoSequ::getDepthRanges() {
    depthFar = vector<double>(nrStereoConfs);
    depthMid = vector<double>(nrStereoConfs);
    depthNear = vector<double>(nrStereoConfs);
    for (size_t i = 0; i < nrStereoConfs; i++) {
        Mat x1, x2;
        if (abs(t[i].at<double>(0)) > abs(t[i].at<double>(1))) {
            if (t[i].at<double>(0) < t[i].at<double>(1)) {
                x1 = (Mat_<double>(3, 1) << (double) imgSize.width, (double) imgSize.height / 2.0, 1.0);
                x2 = (Mat_<double>(3, 1) << 0, (double) imgSize.height / 2.0, 1.0);
            } else {
                x2 = (Mat_<double>(3, 1) << (double) imgSize.width, (double) imgSize.height / 2.0, 1.0);
                x1 = (Mat_<double>(3, 1) << 0, (double) imgSize.height / 2.0, 1.0);
            }
        } else {
            if (t[i].at<double>(1) < t[i].at<double>(0)) {
                x1 = (Mat_<double>(3, 1) << (double) imgSize.width / 2.0, (double) imgSize.height, 1.0);
                x2 = (Mat_<double>(3, 1) << (double) imgSize.width / 2.0, 0, 1.0);
            } else {
                x2 = (Mat_<double>(3, 1) << (double) imgSize.width / 2.0, (double) imgSize.height, 1.0);
                x1 = (Mat_<double>(3, 1) << (double) imgSize.width / 2.0, 0, 1.0);
            }
        }

        double bl = norm(t[i]);
        depthFar[i] = sqrt(K1.at<double>(0, 0) * bl * bl /
                           0.15);//0.15 corresponds to the approx. typ. correspondence accuracy in pixels

        //Calculate min distance for 3D points visible in both images
        Mat b1 = getLineCam1(K1, x1);
        Mat a2, b2;
        getLineCam2(R[i], t[i], K2, x2, a2, b2);
        depthNear[i] = getLineIntersect(b1, a2, b2);
        depthNear[i] = depthNear[i] > 0 ? depthNear[i] : 0;
        depthMid[i] = (depthFar[i] - depthNear[i]) / 2.0;
        if (depthMid[i] < 0) {
            return false;
        }
    }

    return true;
}

/* As the user can specify portions of different depths (near, mid, far) globally for the whole image and also for regions within the image,
these fractions typically do not match. As a result, the depth range fractions per region must be adapted to match the overall fractions of the
whole image. Moreover, the fraction of correspondences per region have an impact on the effective depth portions that must be considered when
adapting the fractions in the image regions.
*/
void genStereoSequ::adaptDepthsPerRegion() {
    if (pars.depthsPerRegion.empty()) {
        pars.depthsPerRegion = std::vector<std::vector<depthPortion>>(3, std::vector<depthPortion>(3));
        std::uniform_real_distribution<double> distribution(0, 1.0);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                pars.depthsPerRegion[i][j] = depthPortion(distribution(rand_gen), distribution(rand_gen),
                                                          distribution(rand_gen));
            }
        }
    } else {
        //Check if the sum of fractions is 1.0
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                pars.depthsPerRegion[i][j].sumTo1();
            }
        }
    }
    pars.corrsPerDepth.sumTo1();

    depthsPerRegion = std::vector<std::vector<std::vector<depthPortion>>>(pars.corrsPerRegion.size(),
                                                                          pars.depthsPerRegion);

    //Correct the portion of depths per region so that they meet the global depth range requirement per image
    for (size_t k = 0; k < pars.corrsPerRegion.size(); k++) {
        //Adapt the fractions of near depths of every region to match the global requirement of the near depth fraction
        updDepthReg(true, depthsPerRegion[k], pars.corrsPerRegion[k]);

        //Update the mid and far depth fractions of each region according to the new near depth fractions
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                double splitrem = 1.0 - depthsPerRegion[k][i][j].near;
                if (!nearZero(splitrem)) {
                    if (!nearZero(depthsPerRegion[k][i][j].mid) && !nearZero(depthsPerRegion[k][i][j].far)) {
                        double fmsum = depthsPerRegion[k][i][j].mid + depthsPerRegion[k][i][j].far;
                        depthsPerRegion[k][i][j].mid = splitrem * depthsPerRegion[k][i][j].mid / fmsum;
                        depthsPerRegion[k][i][j].far = splitrem * depthsPerRegion[k][i][j].far / fmsum;
                    } else if (nearZero(depthsPerRegion[k][i][j].mid) && nearZero(depthsPerRegion[k][i][j].far)) {
                        depthsPerRegion[k][i][j].mid = splitrem / 2.0;
                        depthsPerRegion[k][i][j].far = splitrem / 2.0;
                    } else if (nearZero(depthsPerRegion[k][i][j].mid)) {
                        depthsPerRegion[k][i][j].far = splitrem;
                    } else {
                        depthsPerRegion[k][i][j].mid = splitrem;
                    }
                } else {
                    depthsPerRegion[k][i][j].mid = 0;
                    depthsPerRegion[k][i][j].far = 0;
                }
            }
        }

        //Adapt the fractions of far depths of every region to match the global requirement of the far depth fraction
        updDepthReg(false, depthsPerRegion[k], pars.corrsPerRegion[k]);

        //Update the mid depth fractions of each region according to the new near & far depth fractions
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                depthsPerRegion[k][i][j].mid = 1.0 - (depthsPerRegion[k][i][j].near + depthsPerRegion[k][i][j].far);
            }
        }

#if 1
        //Now, the sum of mid depth regions should correspond to the global requirement
        if(verbose & PRINT_WARNING_MESSAGES) {
            double portSum = 0;
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {
                    portSum += depthsPerRegion[k][i][j].mid * pars.corrsPerRegion[k].at<double>(i, j);
                }
            }
            double c1 = pars.corrsPerDepth.mid - portSum;
            if (!nearZero(c1 / 10.0)) {
                cout << "Adaption of depth fractions in regions failed!" << endl;
            }
        }
#endif
    }
}

//Only adapt the fraction of near or far depths per region to the global requirement
void genStereoSequ::updDepthReg(bool isNear, std::vector<std::vector<depthPortion>> &depthPerRegion, cv::Mat &cpr) {
    //If isNear=false, it is assumed that the fractions of near depths are already fixed
    std::vector<std::vector<double>> oneDepthPerRegion(3, std::vector<double>(3));
    std::vector<std::vector<double>> oneDepthPerRegionMaxVal(3, std::vector<double>(3, 1.0));
    if (isNear) {
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                oneDepthPerRegion[i][j] = depthPerRegion[i][j].near;
            }
        }
    } else {
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                oneDepthPerRegion[i][j] = depthPerRegion[i][j].far;
                oneDepthPerRegionMaxVal[i][j] = 1.0 - depthPerRegion[i][j].near;
            }
        }
    }

    double portSum = 0, c1 = 1.0, dsum = 0, dsum1 = 0;
    size_t cnt = 0;
    //Mat cpr = pars.corrsPerRegion[k];
    while (!nearZero(c1)) {
        cnt++;
        portSum = 0;
        dsum = 0;
        dsum1 = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                portSum += oneDepthPerRegion[i][j] * cpr.at<double>(i, j);
                dsum += oneDepthPerRegion[i][j];
                dsum1 += 1.0 - oneDepthPerRegion[i][j];
            }
        }
        if (isNear)
            c1 = pars.corrsPerDepth.near - portSum;
        else
            c1 = pars.corrsPerDepth.far - portSum;

        bool breakit = false;
        if (!nearZero(c1)) {
            double c12 = 0, c1sum = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    double newval;
                    if (cnt < 3) {
                        CV_Assert(!nearZero(dsum));
                        newval = oneDepthPerRegion[i][j] + c1 * cpr.at<double>(i, j) * oneDepthPerRegion[i][j] / dsum;
                    } else {
                        CV_Assert(!nearZero(dsum) && !nearZero(dsum1));
                        c12 = c1 * cpr.at<double>(i, j) *
                              (0.75 * oneDepthPerRegion[i][j] / dsum + 0.25 * (1.0 - oneDepthPerRegion[i][j]) / dsum1);
                        double c1diff = c1 - (c1sum + c12);
                        if (((c1 > 0) && (c1diff < 0)) ||
                                ((c1 < 0) && (c1diff > 0))) {
                            c12 = c1 - c1sum;
                        }
                        newval = oneDepthPerRegion[i][j] + c12;
                    }
                    if (newval > oneDepthPerRegionMaxVal[i][j]) {
                        c1sum += oneDepthPerRegionMaxVal[i][j] - oneDepthPerRegion[i][j];
                        oneDepthPerRegion[i][j] = oneDepthPerRegionMaxVal[i][j];

                    } else if (newval < 0) {
                        c1sum -= oneDepthPerRegion[i][j];
                        oneDepthPerRegion[i][j] = 0;
                    } else {
                        c1sum += newval - oneDepthPerRegion[i][j];
                        oneDepthPerRegion[i][j] = newval;
                    }
                    if (nearZero(c1sum - c1)) {
                        breakit = true;
                        break;
                    }
                }
                if (breakit) break;
            }
            if (breakit) break;
        }
    }

    if (isNear) {
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                depthPerRegion[i][j].near = oneDepthPerRegion[i][j];
            }
        }
    } else {
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                depthPerRegion[i][j].far = oneDepthPerRegion[i][j];
            }
        }
    }
}

//Check if the given ranges of connected depth areas per image region are correct and initialize them for every definition of depths per image region
void genStereoSequ::checkDepthAreas() {
    //
    //Below: 9 is the nr of regions, minDArea is the min area and 2*sqrt(minDArea) is the gap between areas;
    //size_t maxElems = imgSize.area() / (9 * ((size_t)minDArea + 2 * (size_t)sqrt(minDArea)));
    //Below: 9 is the nr of regions; 4 * (minDArea + sqrt(minDArea)) + 1 corresponds to the area using the side length 2*sqrt(minDArea)+1
    size_t maxElems = (size_t) std::max(imgSize.area() / (9 * (int) (4 * (minDArea + sqrt(minDArea)) + 1)), 1);
    if (pars.nrDepthAreasPReg.empty()) {
        pars.nrDepthAreasPReg = std::vector<std::vector<std::pair<size_t, size_t>>>(3,
                                                                                    std::vector<std::pair<size_t, size_t>>(
                                                                                            3));
        std::uniform_int_distribution<size_t> distribution(1, maxElems + 1);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                size_t tmp = distribution(rand_gen);
                tmp = tmp < 2 ? 2 : tmp;
                size_t tmp1 = distribution(rand_gen) % tmp;
                tmp1 = tmp1 == 0 ? 1 : tmp1;
                pars.nrDepthAreasPReg[i][j] = make_pair(tmp1, tmp);
            }
        }
    } else {
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                if (pars.nrDepthAreasPReg[i][j].first == 0) {
                    pars.nrDepthAreasPReg[i][j].first = 1;
                } else if (pars.nrDepthAreasPReg[i][j].first > (maxElems - 1)) {
                    pars.nrDepthAreasPReg[i][j].first = maxElems - 1;
                }

                if (pars.nrDepthAreasPReg[i][j].second == 0) {
                    pars.nrDepthAreasPReg[i][j].second = 1;
                } else if (pars.nrDepthAreasPReg[i][j].second > maxElems) {
                    pars.nrDepthAreasPReg[i][j].second = maxElems;
                }

                if (pars.nrDepthAreasPReg[i][j].second < pars.nrDepthAreasPReg[i][j].first) {
                    size_t tmp = pars.nrDepthAreasPReg[i][j].first;
                    pars.nrDepthAreasPReg[i][j].first = pars.nrDepthAreasPReg[i][j].second;
                    pars.nrDepthAreasPReg[i][j].second = tmp;
                }
            }
        }
    }

    //Initialize the numbers for every region and depth definition
    nrDepthAreasPRegNear = std::vector<cv::Mat>(depthsPerRegion.size());
    nrDepthAreasPRegMid = std::vector<cv::Mat>(depthsPerRegion.size());
    nrDepthAreasPRegFar = std::vector<cv::Mat>(depthsPerRegion.size());
    for (size_t i = 0; i < depthsPerRegion.size(); i++) {
        nrDepthAreasPRegNear[i] = Mat::ones(3, 3, CV_32SC1);
        nrDepthAreasPRegMid[i] = Mat::ones(3, 3, CV_32SC1);
        nrDepthAreasPRegFar[i] = Mat::ones(3, 3, CV_32SC1);
    }

    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (pars.nrDepthAreasPReg[y][x].second < 4) {
                for (size_t i = 0; i < depthsPerRegion.size(); i++) {
                    if (!nearZero(depthsPerRegion[i][y][x].near) &&
                        !nearZero(depthsPerRegion[i][y][x].mid) &&
                        !nearZero(depthsPerRegion[i][y][x].far)) {
                        continue;//1 remains in every element
                    } else {
                        int cnt = (int) pars.nrDepthAreasPReg[y][x].second;
                        int tmp = -10;
                        nrDepthAreasPRegNear[i].at<int32_t>(y, x) = 0;
                        nrDepthAreasPRegMid[i].at<int32_t>(y, x) = 0;
                        nrDepthAreasPRegFar[i].at<int32_t>(y, x) = 0;
                        bool lockdistr[3] = {true, true, true};
                        while (cnt > 0) {
                            if (!nearZero(depthsPerRegion[i][y][x].near) && lockdistr[0]) {
                                cnt--;
                                nrDepthAreasPRegNear[i].at<int32_t>(y, x)++;
                            }
                            if (!nearZero(depthsPerRegion[i][y][x].mid) && lockdistr[1]) {
                                cnt--;
                                nrDepthAreasPRegMid[i].at<int32_t>(y, x)++;
                            }
                            if (!nearZero(depthsPerRegion[i][y][x].far) && lockdistr[2]) {
                                cnt--;
                                nrDepthAreasPRegFar[i].at<int32_t>(y, x)++;
                            }
                            if ((cnt > 0) && (tmp == -10)) {
                                if ((pars.nrDepthAreasPReg[y][x].second - pars.nrDepthAreasPReg[y][x].first) != 0) {
                                    tmp = cnt - (int) pars.nrDepthAreasPReg[y][x].second;
                                    tmp += (int)(pars.nrDepthAreasPReg[y][x].first + (rand2() %
                                                                                (pars.nrDepthAreasPReg[y][x].second -
                                                                                 pars.nrDepthAreasPReg[y][x].first +
                                                                                 1)));
                                    cnt = tmp;
                                }
                                if (cnt > 0) {
                                    if (!(!nearZero(depthsPerRegion[i][y][x].near) &&
                                          ((depthsPerRegion[i][y][x].near > depthsPerRegion[i][y][x].mid) ||
                                           (depthsPerRegion[i][y][x].near > depthsPerRegion[i][y][x].far)))) {
                                        lockdistr[0] = false;
                                    }
                                    if (!(!nearZero(depthsPerRegion[i][y][x].mid) &&
                                          ((depthsPerRegion[i][y][x].mid > depthsPerRegion[i][y][x].near) ||
                                           (depthsPerRegion[i][y][x].mid > depthsPerRegion[i][y][x].far)))) {
                                        lockdistr[1] = false;
                                    }
                                    if (!(!nearZero(depthsPerRegion[i][y][x].far) &&
                                          ((depthsPerRegion[i][y][x].far > depthsPerRegion[i][y][x].near) ||
                                           (depthsPerRegion[i][y][x].far > depthsPerRegion[i][y][x].mid)))) {
                                        lockdistr[2] = false;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < depthsPerRegion.size(); i++) {
                    int nra = (int)pars.nrDepthAreasPReg[y][x].first + ((int)(rand2() % INT_MAX) % ((int)pars.nrDepthAreasPReg[y][x].second -
                            (int)pars.nrDepthAreasPReg[y][x].first +
                                                                                  1));
                    int32_t maxAPReg[3];
                    double maxAPRegd[3];
                    maxAPRegd[0] = depthsPerRegion[i][y][x].near * (double) nra;
                    maxAPRegd[1] = depthsPerRegion[i][y][x].mid * (double) nra;
                    maxAPRegd[2] = depthsPerRegion[i][y][x].far * (double) nra;
                    maxAPReg[0] = (int32_t) round(maxAPRegd[0]);
                    maxAPReg[1] = (int32_t) round(maxAPRegd[1]);
                    maxAPReg[2] = (int32_t) round(maxAPRegd[2]);
                    int32_t diffap = (int32_t) nra - (maxAPReg[0] + maxAPReg[1] + maxAPReg[2]);
                    if (diffap != 0) {
                        maxAPRegd[0] -= (double) maxAPReg[0];
                        maxAPRegd[1] -= (double) maxAPReg[1];
                        maxAPRegd[2] -= (double) maxAPReg[2];
                        if (diffap < 0) {
                            int cnt = 0;
                            std::ptrdiff_t pdiff = min_element(maxAPRegd, maxAPRegd + 3) - maxAPRegd;
                            while ((diffap < 0) && (cnt < 3)) {
                                if (maxAPReg[pdiff] > 1) {
                                    maxAPReg[pdiff]--;
                                    diffap++;
                                }
                                if (diffap < 0) {
                                    if ((maxAPReg[(pdiff + 1) % 3] > 1) &&
                                        (maxAPRegd[(pdiff + 1) % 3] <= maxAPRegd[(pdiff + 2) % 3])) {
                                        maxAPReg[(pdiff + 1) % 3]--;
                                        diffap++;
                                    } else if ((maxAPReg[(pdiff + 2) % 3] > 1) &&
                                               (maxAPRegd[(pdiff + 2) % 3] < maxAPRegd[(pdiff + 1) % 3])) {
                                        maxAPReg[(pdiff + 2) % 3]--;
                                        diffap++;
                                    }
                                }
                                cnt++;
                            }
                        } else {
                            std::ptrdiff_t pdiff = max_element(maxAPRegd, maxAPRegd + 3) - maxAPRegd;
                            while (diffap > 0) {
                                maxAPReg[pdiff]++;
                                diffap--;
                                if (diffap > 0) {
                                    if (maxAPRegd[(pdiff + 1) % 3] >= maxAPRegd[(pdiff + 2) % 3]) {
                                        maxAPReg[(pdiff + 1) % 3]++;
                                        diffap--;
                                    } else {
                                        maxAPReg[(pdiff + 2) % 3]++;
                                        diffap--;
                                    }
                                }
                            }
                        }
                    }

                    nrDepthAreasPRegNear[i].at<int32_t>(y, x) = maxAPReg[0];
                    nrDepthAreasPRegMid[i].at<int32_t>(y, x) = maxAPReg[1];
                    nrDepthAreasPRegFar[i].at<int32_t>(y, x) = maxAPReg[2];
                    if (!nearZero(depthsPerRegion[i][y][x].near) && (maxAPReg[0] == 0)) {
                        nrDepthAreasPRegNear[i].at<int32_t>(y, x)++;
                    }
                    if (!nearZero(depthsPerRegion[i][y][x].mid) && (maxAPReg[1] == 0)) {
                        nrDepthAreasPRegMid[i].at<int32_t>(y, x)++;
                    }
                    if (!nearZero(depthsPerRegion[i][y][x].far) && (maxAPReg[2] == 0)) {
                        nrDepthAreasPRegFar[i].at<int32_t>(y, x)++;
                    }
                }
            }
        }
    }
}

//Calculate the area in pixels for every depth and region
void genStereoSequ::calcPixAreaPerDepth() {
//    int32_t regArea = (int32_t) imgSize.area() / 9;
    areaPRegNear.resize(depthsPerRegion.size());
    areaPRegMid.resize(depthsPerRegion.size());
    areaPRegFar.resize(depthsPerRegion.size());

    for (size_t i = 0; i < depthsPerRegion.size(); i++) {
        areaPRegNear[i] = Mat::zeros(3, 3, CV_32SC1);
        areaPRegMid[i] = Mat::zeros(3, 3, CV_32SC1);
        areaPRegFar[i] = Mat::zeros(3, 3, CV_32SC1);
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                int32_t tmp[3] = {0, 0, 0};
                int regArea = regROIs[y][x].area();
                tmp[0] = (int32_t) round(depthsPerRegion[i][y][x].near * (double) regArea);
                if ((tmp[0] != 0) && (tmp[0] < minDArea))
                    tmp[0] = minDArea;

                tmp[1] = (int32_t) round(depthsPerRegion[i][y][x].mid * (double) regArea);
                if ((tmp[1] != 0) && (tmp[1] < minDArea))
                    tmp[1] = minDArea;

                tmp[2] = (int32_t) round(depthsPerRegion[i][y][x].far * (double) regArea);
                if ((tmp[2] != 0) && (tmp[2] < minDArea))
                    tmp[2] = minDArea;

                if ((tmp[0] + tmp[1] + tmp[2]) != regArea) {
                    std::ptrdiff_t pdiff = max_element(tmp, tmp + 3) - tmp;
                    tmp[pdiff] = regArea - tmp[(pdiff + 1) % 3] - tmp[(pdiff + 2) % 3];
                }

                areaPRegNear[i].at<int32_t>(y, x) = tmp[0];
                areaPRegMid[i].at<int32_t>(y, x) = tmp[1];
                areaPRegFar[i].at<int32_t>(y, x) = tmp[2];
            }
        }
    }
}

/*Backproject 3D points (generated one or more frames before) found to be possibly visible in the
current stereo rig position to the stereo image planes and check if they are visible or produce
outliers in the first or second stereo image.
*/
void genStereoSequ::backProject3D() {
    if (!actCorrsImg2TNFromLast.empty())
        actCorrsImg2TNFromLast.release();
    if (!actCorrsImg2TNFromLast_Idx.empty())
        actCorrsImg2TNFromLast_Idx.clear();
    if (!actCorrsImg1TNFromLast.empty())
        actCorrsImg1TNFromLast.release();
    if (!actCorrsImg1TNFromLast_Idx.empty())
        actCorrsImg1TNFromLast_Idx.clear();
    if (!actCorrsImg1TPFromLast.empty())
        actCorrsImg1TPFromLast.release();
    if (!actCorrsImg2TPFromLast.empty())
        actCorrsImg2TPFromLast.release();
    if (!actCorrsImg12TPFromLast_Idx.empty())
        actCorrsImg12TPFromLast_Idx.clear();

    if (actImgPointCloudFromLast.empty())
        return;

    struct imgWH {
        double width;
        double height;
        double maxDist;
    } dimgWH = {0,0,0};
    dimgWH.width = (double) (imgSize.width - 1);
    dimgWH.height = (double) (imgSize.height - 1);
    dimgWH.maxDist = maxFarDistMultiplier * actDepthFar;

    std::vector<cv::Point3d> actImgPointCloudFromLast_tmp;
    vector<int> actCorrsImg12TPFromLast_IdxWorld_tmp;
    size_t idx1 = 0;
    for (auto i = 0; i < (int)actImgPointCloudFromLast.size(); ++i) {
        cv::Point3d &pt = actImgPointCloudFromLast[i];
        if ((pt.z < actDepthNear) ||
            (pt.z > dimgWH.maxDist)) {
            continue;
        }

        Mat X = Mat(pt, false).reshape(1, 3);
        Mat x1 = K1 * X;
        if(nearZero(x1.at<double>(2))) continue;
        x1 /= x1.at<double>(2);

        bool outOfR[2] = {false, false};

        //Check if the point is within the area of a moving object
        Point x1r = Point((int) round(x1.at<double>(0)), (int) round(x1.at<double>(1)));
        if ((x1r.x < 0) || (x1r.x > dimgWH.width) || (x1r.y < 0) || (x1r.y > dimgWH.height)) {
            outOfR[0] = true;
        } else if (combMovObjLabelsAll.at<unsigned char>(x1r.y, x1r.x) > 0)
            continue;

        if ((x1.at<double>(0) < 0) || (x1.at<double>(0) > dimgWH.width) ||
            (x1.at<double>(1) < 0) || (x1.at<double>(1) > dimgWH.height))//Not visible in first image
        {
            outOfR[0] = true;
        }

        Mat x2 = K2 * (actR * X + actT);
        if(nearZero(x2.at<double>(2))) continue;
        x2 /= x2.at<double>(2);

        if ((x2.at<double>(0) < 0) || (x2.at<double>(0) > dimgWH.width) ||
            (x2.at<double>(1) < 0) || (x2.at<double>(1) > dimgWH.height))//Not visible in second image
        {
            outOfR[1] = true;
        }

        //Check if the point is within the area of a moving object in the second image
        Point x2r = Point((int) round(x2.at<double>(0)), (int) round(x2.at<double>(1)));
        if ((x2r.x < 0) || (x2r.x > dimgWH.width) || (x2r.y < 0) || (x2r.y > dimgWH.height)) {
            outOfR[1] = true;
        } else if (movObjMask2All.at<unsigned char>(x2r.y, x2r.x) > 0)
            outOfR[1] = true;

        if (outOfR[0] && outOfR[1]) {
            continue;
        } else if (outOfR[0]) {
            actCorrsImg2TNFromLast.push_back(x2.t());
            actCorrsImg2TNFromLast_Idx.push_back(idx1);
        } else if (outOfR[1]) {
            actCorrsImg1TNFromLast.push_back(x1.t());
            actCorrsImg1TNFromLast_Idx.push_back(idx1);
        } else {
            actCorrsImg1TPFromLast.push_back(x1.t());
            actCorrsImg2TPFromLast.push_back(x2.t());
            actCorrsImg12TPFromLast_Idx.push_back(idx1);
        }
        actImgPointCloudFromLast_tmp.push_back(pt);
        actCorrsImg12TPFromLast_IdxWorld_tmp.push_back(actCorrsImg12TPFromLast_IdxWorld[i]);
        idx1++;
    }
    actImgPointCloudFromLast = actImgPointCloudFromLast_tmp;
    actCorrsImg12TPFromLast_IdxWorld = actCorrsImg12TPFromLast_IdxWorld_tmp;
    if (!actCorrsImg1TNFromLast.empty())
        actCorrsImg1TNFromLast = actCorrsImg1TNFromLast.t();
    if (!actCorrsImg2TNFromLast.empty())
        actCorrsImg2TNFromLast = actCorrsImg2TNFromLast.t();
    if (!actCorrsImg1TPFromLast.empty()) {
        actCorrsImg1TPFromLast = actCorrsImg1TPFromLast.t();
        actCorrsImg2TPFromLast = actCorrsImg2TPFromLast.t();
    }
}

//Generate seeds for generating depth areas and include the seeds found by backprojection of the 3D points of the last frames
void genStereoSequ::checkDepthSeeds() {
    seedsNear = std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>>(3,
                                                                            std::vector<std::vector<cv::Point3_<int32_t>>>(
                                                                                    3));
    seedsMid = std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>>(3,
                                                                           std::vector<std::vector<cv::Point3_<int32_t>>>(
                                                                                   3));
    seedsFar = std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>>(3,
                                                                           std::vector<std::vector<cv::Point3_<int32_t>>>(
                                                                                   3));

    //Generate a mask for marking used areas in the first stereo image
    corrsIMG = Mat::zeros(imgSize.height + csurr.cols - 1, imgSize.width + csurr.rows - 1, CV_8UC1);

    int posadd1 = max((int) ceil(pars.minKeypDist), (int) sqrt(minDArea));
    int sqrSi1 = 2 * posadd1;
    cv::Mat filtInitPts = Mat::zeros(imgSize.height + sqrSi1, imgSize.width + sqrSi1, CV_8UC1);
    sqrSi1++;
    Mat csurr1 = Mat::ones(sqrSi1, sqrSi1, CV_8UC1);
    //int maxSum1 = sqrSi1 * sqrSi1;

    cv::Size regSi = Size(imgSize.width / 3, imgSize.height / 3);
    if (!actCorrsImg1TPFromLast.empty())//Take seeding positions from backprojected coordinates
    {
        std::vector<cv::Point3_<int32_t>> seedsNear_tmp, seedsNear_tmp1;
        std::vector<cv::Point3_<int32_t>> seedsMid_tmp, seedsMid_tmp1;
        std::vector<cv::Point3_<int32_t>> seedsFar_tmp, seedsFar_tmp1;
        //Identify depth categories
        for (size_t i = 0; i < actCorrsImg12TPFromLast_Idx.size(); i++) {
            if (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[i]].z >= actDepthFar) {
                seedsFar_tmp.emplace_back(cv::Point3_<int32_t>((int32_t) round(actCorrsImg1TPFromLast.at<double>(0, (int)i)),
                                                            (int32_t) round(actCorrsImg1TPFromLast.at<double>(1, (int)i)),
                                                            (int32_t) i));
            } else if (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[i]].z >= actDepthMid) {
                seedsMid_tmp.emplace_back(cv::Point3_<int32_t>((int32_t) round(actCorrsImg1TPFromLast.at<double>(0, (int)i)),
                                                            (int32_t) round(actCorrsImg1TPFromLast.at<double>(1, (int)i)),
                                                            (int32_t) i));
            } else {
                seedsNear_tmp.emplace_back(cv::Point3_<int32_t>((int32_t) round(actCorrsImg1TPFromLast.at<double>(0, (int)i)),
                                                             (int32_t) round(actCorrsImg1TPFromLast.at<double>(1, (int)i)),
                                                             (int32_t) i));
            }
        }

        //Check if the seeds are too near to each other
        int posadd = max((int) ceil(pars.minKeypDist), 1);
        int sqrSi = 2 * posadd;
        //cv::Mat filtInitPts = Mat::zeros(imgSize.width + sqrSi, imgSize.height + sqrSi, CV_8UC1);
        sqrSi++;//sqrSi = 2 * (int)floor(pars.minKeypDist) + 1;
        //csurr = Mat::ones(sqrSi, sqrSi, CV_8UC1);
        //int maxSum = sqrSi * sqrSi;
        int sqrSiDiff2 = (sqrSi1 - sqrSi) / 2;
        int hlp2 = sqrSi + sqrSiDiff2;

        vector<size_t> delListCorrs, delList3D;
        if (!seedsNear_tmp.empty()) {
            for (auto& i : seedsNear_tmp) {
                Mat s_tmp = filtInitPts(Range(i.y + sqrSiDiff2, i.y + hlp2),
                                        Range(i.x + sqrSiDiff2, i.x + hlp2));
                if (s_tmp.at<unsigned char>(posadd, posadd) > 0) {
                    delListCorrs.push_back((size_t) i.z);
                    delList3D.push_back(actCorrsImg12TPFromLast_Idx[delListCorrs.back()]);
                    continue;
                }
//                csurr.copyTo(s_tmp);
                s_tmp += csurr;
                seedsNear_tmp1.push_back(i);
            }
        }
        if (!seedsMid_tmp.empty()) {
            for (auto& i : seedsMid_tmp) {
                Mat s_tmp = filtInitPts(Range(i.y + sqrSiDiff2, i.y + hlp2),
                                        Range(i.x + sqrSiDiff2, i.x + hlp2));
                if (s_tmp.at<unsigned char>(posadd, posadd) > 0) {
                    delListCorrs.push_back((size_t) i.z);
                    delList3D.push_back(actCorrsImg12TPFromLast_Idx[delListCorrs.back()]);
                    continue;
                }
//                csurr.copyTo(s_tmp);
                s_tmp += csurr;
                seedsMid_tmp1.push_back(i);
            }
        }
        if (!seedsFar_tmp.empty()) {
            for (auto& i : seedsFar_tmp) {
                Mat s_tmp = filtInitPts(Range(i.y + sqrSiDiff2, i.y + hlp2),
                                        Range(i.x + sqrSiDiff2, i.x + hlp2));
                if (s_tmp.at<unsigned char>(posadd, posadd) > 0) {
                    delListCorrs.push_back((size_t) i.z);
                    delList3D.push_back(actCorrsImg12TPFromLast_Idx[delListCorrs.back()]);
                    continue;
                }
//                csurr.copyTo(s_tmp);
                s_tmp += csurr;
                seedsFar_tmp1.push_back(i);
            }
        }
        filtInitPts(Rect(sqrSiDiff2, sqrSiDiff2, imgSize.width + 2 * posadd, imgSize.height + 2 * posadd)).copyTo(
                corrsIMG);

        //Check if seedsNear only holds near distances
        /*for (int j = 0; j < seedsNear_tmp1.size(); ++j) {
            if(actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[seedsNear_tmp1[j].z]].z >= actDepthMid){
                cout << "Wrong distance!" << endl;
            }
        }
        //Check if seedsMid only holds mid distances
        for (int j = 0; j < seedsMid_tmp1.size(); ++j) {
            if((actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[seedsMid_tmp1[j].z]].z < actDepthMid) ||
               (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[seedsMid_tmp1[j].z]].z >= actDepthFar)){
                cout << "Wrong distance!" << endl;
            }
        }
        //Check if seedsFar only holds far distances
        for (int j = 0; j < seedsFar_tmp1.size(); ++j) {
            if(actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[seedsFar_tmp1[j].z]].z < actDepthFar){
                cout << "Wrong distance!" << endl;
            }
        }*/

        //Delete correspondences and 3D points that were to near to each other in the image
        if (!delListCorrs.empty()) {
//            if(!checkCorr3DConsistency()){
//                throw SequenceException("Correspondences are not projections of 3D points!");
//            }
            /*std::vector<cv::Point3d> actImgPointCloudFromLast_tmp;
            cv::Mat actCorrsImg1TPFromLast_tmp, actCorrsImg2TPFromLast_tmp;*/

            sort(delList3D.begin(), delList3D.end(),
                 [](size_t first, size_t second) { return first < second; });//Ascending order

            if (!actCorrsImg1TNFromLast_Idx.empty())//Adapt the indices for TN (single keypoints without a match)
            {
                adaptIndicesNoDel(actCorrsImg1TNFromLast_Idx, delList3D);
            }
            if (!actCorrsImg2TNFromLast_Idx.empty())//Adapt the indices for TN (single keypoints without a match)
            {
                adaptIndicesNoDel(actCorrsImg2TNFromLast_Idx, delList3D);
            }
            adaptIndicesNoDel(actCorrsImg12TPFromLast_Idx, delList3D);
            deleteVecEntriesbyIdx(actImgPointCloudFromLast, delList3D);
            deleteVecEntriesbyIdx(actCorrsImg12TPFromLast_IdxWorld, delList3D);

            sort(delListCorrs.begin(), delListCorrs.end(), [](size_t first, size_t second) { return first < second; });
            if (!seedsNear_tmp1.empty())
                adaptIndicesCVPtNoDel(seedsNear_tmp1, delListCorrs);
            if (!seedsMid_tmp1.empty())
                adaptIndicesCVPtNoDel(seedsMid_tmp1, delListCorrs);
            if (!seedsFar_tmp1.empty())
                adaptIndicesCVPtNoDel(seedsFar_tmp1, delListCorrs);
            deleteVecEntriesbyIdx(actCorrsImg12TPFromLast_Idx, delListCorrs);
            deleteMatEntriesByIdx(actCorrsImg1TPFromLast, delListCorrs, false);
            deleteMatEntriesByIdx(actCorrsImg2TPFromLast, delListCorrs, false);
//            if(!checkCorr3DConsistency()){
//                throw SequenceException("Correspondences are not projections of 3D points!");
//            }
        }

        //Add the seeds to their regions
        for (auto& i : seedsNear_tmp1) {
            int32_t ix = i.x / regSi.width;
            ix = (ix > 2) ? 2 : ix;
            int32_t iy = i.y / regSi.height;
            iy = (iy > 2) ? 2 : iy;
            seedsNear[iy][ix].push_back(i);
        }

        //Add the seeds to their regions
        for (auto& i : seedsMid_tmp1) {
            int32_t ix = i.x / regSi.width;
            ix = (ix > 2) ? 2 : ix;
            int32_t iy = i.y / regSi.height;
            iy = (iy > 2) ? 2 : iy;
            seedsMid[iy][ix].push_back(i);
        }

        //Add the seeds to their regions
        for (auto& i : seedsFar_tmp1) {
            int32_t ix = i.x / regSi.width;
            ix = (ix > 2) ? 2 : ix;
            int32_t iy = i.y / regSi.height;
            iy = (iy > 2) ? 2 : iy;
            seedsFar[iy][ix].push_back(i);
        }
    }

    //Check if seedsNear only holds near distances
    for (int l = 0; l < 3; ++l) {
        for (int i = 0; i < 3; ++i) {
            for (auto& j : seedsNear[l][i]) {
                if (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[j.z]].z >= actDepthMid) {
                    throw SequenceException(
                            "Seeding depth of backprojected static 3D points in the category near is out of range!");
                }
            }
        }
    }
    //Check if seedsMid only holds mid distances
    for (int l = 0; l < 3; ++l) {
        for (int i = 0; i < 3; ++i) {
            for (auto& j : seedsMid[l][i]) {
                if ((actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[j.z]].z < actDepthMid) ||
                    (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[j.z]].z >= actDepthFar)) {
                    throw SequenceException(
                            "Seeding depth of backprojected static 3D points in the category mid is out of range!");
                }
            }
        }
    }
    //Check if seedsFar only holds far distances
    for (int l = 0; l < 3; ++l) {
        for (int i = 0; i < 3; ++i) {
            for (auto& j : seedsFar[l][i]) {
                if (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[j.z]].z < actDepthFar) {
                    throw SequenceException(
                            "Seeding depth of backprojected static 3D points in the category far is out of range!");
                }
            }
        }
    }

    //Generate new seeds
    Point3_<int32_t> pt;
    pt.z = -1;
    for (int32_t y = 0; y < 3; y++) {
        int32_t mmy[2];
        mmy[0] = y * regSi.height;
        mmy[1] = mmy[0] + regSi.height - 1;
        std::uniform_int_distribution<int32_t> distributionY(mmy[0], mmy[1]);
        for (int32_t x = 0; x < 3; x++) {
            int32_t mmx[2];
            mmx[0] = x * regSi.width;
            mmx[1] = mmx[0] + regSi.width - 1;
            std::uniform_int_distribution<int32_t> distributionX(mmx[0], mmx[1]);
            int32_t diffNr = nrDepthAreasPRegNear[actCorrsPRIdx].at<int32_t>(y, x) - (int32_t) seedsNear[y][x].size();
            uint32_t it_cnt = 0;
            const uint32_t max_it_cnt = 500000;
            while ((diffNr > 0) && (it_cnt < max_it_cnt))//Generate seeds for near depth areas
            {
                pt.x = distributionX(rand_gen);
                pt.y = distributionY(rand_gen);
                Mat s_tmp = filtInitPts(Range(pt.y, pt.y + sqrSi1), Range(pt.x, pt.x + sqrSi1));
                if (s_tmp.at<unsigned char>(posadd1, posadd1) > 0) {
                    it_cnt++;
                    continue;
                } else {
                    csurr1.copyTo(s_tmp);
                    seedsNear[y][x].push_back(pt);
                    diffNr--;
                }
            }
            if (diffNr > 0){
                nrDepthAreasPRegNear[actCorrsPRIdx].at<int32_t>(y, x) -= diffNr;
                cout << "Unable to reach desired number of near depth areas in region ("
                << y << ", " << x << "). " << diffNr << " areas cannot be assigned." << endl;
            }
            it_cnt = 0;
            diffNr = nrDepthAreasPRegMid[actCorrsPRIdx].at<int32_t>(y, x) - (int32_t) seedsMid[y][x].size();
            while ((diffNr > 0) && (it_cnt < max_it_cnt))//Generate seeds for mid depth areas
            {
                pt.x = distributionX(rand_gen);
                pt.y = distributionY(rand_gen);
                Mat s_tmp = filtInitPts(Range(pt.y, pt.y + sqrSi1), Range(pt.x, pt.x + sqrSi1));
                if (s_tmp.at<unsigned char>(posadd1, posadd1) > 0) {
                    it_cnt++;
                    continue;
                } else {
                    csurr1.copyTo(s_tmp);
                    seedsMid[y][x].push_back(pt);
                    diffNr--;
                }
            }
            if (diffNr > 0){
                nrDepthAreasPRegMid[actCorrsPRIdx].at<int32_t>(y, x) -= diffNr;
                cout << "Unable to reach desired number of mid depth areas in region ("
                     << y << ", " << x << "). " << diffNr << " areas cannot be assigned." << endl;
            }
            it_cnt = 0;
            diffNr = nrDepthAreasPRegFar[actCorrsPRIdx].at<int32_t>(y, x) - (int32_t) seedsFar[y][x].size();
            while ((diffNr > 0) && (it_cnt < max_it_cnt))//Generate seeds for far depth areas
            {
                pt.x = distributionX(rand_gen);
                pt.y = distributionY(rand_gen);
                Mat s_tmp = filtInitPts(Range(pt.y, pt.y + sqrSi1), Range(pt.x, pt.x + sqrSi1));
                if (s_tmp.at<unsigned char>(posadd1, posadd1) > 0) {
                    it_cnt++;
                    continue;
                } else {
                    csurr1.copyTo(s_tmp);
                    seedsFar[y][x].push_back(pt);
                    diffNr--;
                }
            }
            if (diffNr > 0){
                nrDepthAreasPRegFar[actCorrsPRIdx].at<int32_t>(y, x) -= diffNr;
                cout << "Unable to reach desired number of far depth areas in region ("
                     << y << ", " << x << "). " << diffNr << " areas cannot be assigned." << endl;
            }
        }
    }

    //Check if seedsNear only holds near distances
    /*for (int l = 0; l < 3; ++l) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < seedsNear[l][i].size(); ++j) {
                if(seedsNear[l][i][j].z >= 0) {
                    if (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[seedsNear[l][i][j].z]].z >= actDepthMid) {
                        cout << "Wrong distance!" << endl;
                    }
                }
            }
        }
    }
    //Check if seedsMid only holds near distances
    for (int l = 0; l < 3; ++l) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < seedsMid[l][i].size(); ++j) {
                if(seedsMid[l][i][j].z >= 0) {
                    if ((actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[seedsMid[l][i][j].z]].z < actDepthMid) ||
                        (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[seedsMid[l][i][j].z]].z >= actDepthFar)) {
                        cout << "Wrong distance!" << endl;
                    }
                }
            }
        }
    }
    //Check if seedsMid only holds near distances
    for (int l = 0; l < 3; ++l) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < seedsFar[l][i].size(); ++j) {
                if(seedsFar[l][i][j].z >= 0) {
                    if (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[seedsFar[l][i][j].z]].z < actDepthFar) {
                        cout << "Wrong distance!" << endl;
                    }
                }
            }
        }
    }*/

    //Get distances between neighboring seeds
    SeedCloud sc = {nullptr, nullptr, nullptr};
    sc.seedsNear = &seedsNear;
    sc.seedsMid = &seedsMid;
    sc.seedsFar = &seedsFar;
    // construct a kd-tree index:
    typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<int32_t, SeedCloud>,
            SeedCloud,
            2 /* dim */
    > my_kd_tree_t;
    my_kd_tree_t index(2 /*dim*/, sc, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();
    size_t num_results_in = 2, num_results_out = 0;
    std::vector<size_t> ret_index(num_results_in);
    std::vector<int32_t> out_dist_sqr(num_results_in);

    seedsNearNNDist = std::vector<std::vector<std::vector<int32_t>>>(3, std::vector<std::vector<int32_t>>(3));
    for (int32_t j = 0; j < 3; ++j) {
        for (int32_t i = 0; i < 3; ++i) {
            seedsNearNNDist[j][i] = std::vector<int32_t>(seedsNear[j][i].size());
            for (size_t k = 0; k < seedsNear[j][i].size(); ++k) {
                num_results_out = index.knnSearch(&seedsNear[j][i][k].x, num_results_in, &ret_index[0],
                                                  &out_dist_sqr[0]);
                if (num_results_out == num_results_in) {
                    seedsNearNNDist[j][i][k] = (int32_t)floor(sqrt(out_dist_sqr[1]));
                } else {
                    int32_t negdistx = seedsNear[j][i][k].x - i * regSi.width;
                    int32_t posdistx = (i + 1) * regSi.width - seedsNear[j][i][k].x;
                    int32_t negdisty = seedsNear[j][i][k].y - j * regSi.height;
                    int32_t posdisty = (j + 1) * regSi.height - seedsNear[j][i][k].y;
                    seedsNearNNDist[j][i][k] = max(negdistx, max(posdistx, max(negdisty, posdisty)));
                }
            }
        }
    }
    seedsMidNNDist = std::vector<std::vector<std::vector<int32_t>>>(3, std::vector<std::vector<int32_t>>(3));
    for (int32_t j = 0; j < 3; ++j) {
        for (int32_t i = 0; i < 3; ++i) {
            seedsMidNNDist[j][i] = std::vector<int32_t>(seedsMid[j][i].size());
            for (size_t k = 0; k < seedsMid[j][i].size(); ++k) {
                num_results_out = index.knnSearch(&seedsMid[j][i][k].x, num_results_in, &ret_index[0],
                                                  &out_dist_sqr[0]);
                if (num_results_out == num_results_in) {
                    seedsMidNNDist[j][i][k] = (int32_t)floor(sqrt(out_dist_sqr[1]));
                } else {
                    int32_t negdistx = seedsMid[j][i][k].x - i * regSi.width;
                    int32_t posdistx = (i + 1) * regSi.width - seedsMid[j][i][k].x;
                    int32_t negdisty = seedsMid[j][i][k].y - j * regSi.height;
                    int32_t posdisty = (j + 1) * regSi.height - seedsMid[j][i][k].y;
                    seedsMidNNDist[j][i][k] = max(negdistx, max(posdistx, max(negdisty, posdisty)));
                }
            }
        }
    }
    seedsFarNNDist = std::vector<std::vector<std::vector<int32_t>>>(3, std::vector<std::vector<int32_t>>(3));
    for (int32_t j = 0; j < 3; ++j) {
        for (int32_t i = 0; i < 3; ++i) {
            seedsFarNNDist[j][i] = std::vector<int32_t>(seedsFar[j][i].size());
            for (size_t k = 0; k < seedsFar[j][i].size(); ++k) {
                num_results_out = index.knnSearch(&seedsFar[j][i][k].x, num_results_in, &ret_index[0],
                                                  &out_dist_sqr[0]);
                if (num_results_out == num_results_in) {
                    seedsFarNNDist[j][i][k] = (int32_t)floor(sqrt(out_dist_sqr[1]));
                } else {
                    int32_t negdistx = seedsFar[j][i][k].x - i * regSi.width;
                    int32_t posdistx = (i + 1) * regSi.width - seedsFar[j][i][k].x;
                    int32_t negdisty = seedsFar[j][i][k].y - j * regSi.height;
                    int32_t posdisty = (j + 1) * regSi.height - seedsFar[j][i][k].y;
                    seedsFarNNDist[j][i][k] = max(negdistx, max(posdistx, max(negdisty, posdisty)));
                }
            }
        }
    }
}

//Wrapper function for function adaptIndicesNoDel
void genStereoSequ::adaptIndicesCVPtNoDel(std::vector<cv::Point3_<int32_t>> &seedVec,
                                          std::vector<size_t> &delListSortedAsc) {
    std::vector<size_t> seedVecIdx;
    seedVecIdx.reserve(seedVec.size());
    for (auto &sV : seedVec) {
        seedVecIdx.push_back((size_t) sV.z);
    }
    adaptIndicesNoDel(seedVecIdx, delListSortedAsc);
    for (size_t i = 0; i < seedVecIdx.size(); i++) {
        seedVec[i].z = (int32_t)seedVecIdx[i];
    }
}

//Adapt the indices of a not continious vector for which a part of the target data where the indices point to was deleted (no data points the indices point to were deleted).
void genStereoSequ::adaptIndicesNoDel(std::vector<size_t> &idxVec, std::vector<size_t> &delListSortedAsc) {
    std::vector<pair<size_t, size_t>> idxVec_tmp(idxVec.size());
    for (size_t i = 0; i < idxVec.size(); i++) {
        idxVec_tmp[i] = make_pair(idxVec[i], i);
    }
    sort(idxVec_tmp.begin(), idxVec_tmp.end(),
         [](pair<size_t, size_t> first, pair<size_t, size_t> second) { return first.first < second.first; });
    size_t idx = 0;
    size_t maxIdx = delListSortedAsc.size() - 1;
    for (auto& i : idxVec_tmp) {
        if (idx <= maxIdx) {
            if (i.first < delListSortedAsc[idx]) {
                i.first -= idx;
            } else {
                while ((idx <= maxIdx) && (i.first > delListSortedAsc[idx])) {
                    idx++;
                }
                i.first -= idx;
            }
        } else {
            i.first -= idx;
        }
    }
    sort(idxVec_tmp.begin(), idxVec_tmp.end(),
         [](pair<size_t, size_t> first, pair<size_t, size_t> second) { return first.second < second.second; });
    for (size_t i = 0; i < idxVec_tmp.size(); i++) {
        idxVec[i] = idxVec_tmp[i].first;
    }
}

int genStereoSequ::deleteBackProjTPByDepth(std::vector<cv::Point_<int32_t>> &seedsFromLast,
                        int32_t nrToDel){
    std::vector<size_t> delListCorrs, delList3D;
    int actDelNr = deletedepthCatsByNr(seedsFromLast, nrToDel, actCorrsImg1TPFromLast, delListCorrs);

    int32_t kSi = csurr.rows;
    delList3D.reserve(delListCorrs.size());
    for(auto &i: delListCorrs){
        delList3D.push_back(actCorrsImg12TPFromLast_Idx[i]);

        Point pt = Point((int)round(actCorrsImg1TPFromLast.at<double>(0,i)),
                         (int)round(actCorrsImg1TPFromLast.at<double>(1,i)));
        Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
        s_tmp -= csurr;
    }

    if (!delListCorrs.empty()) {
//        if(!checkCorr3DConsistency()){
//            throw SequenceException("Correspondences are not projections of 3D points!");
//        }
        sort(delList3D.begin(), delList3D.end(),
             [](size_t first, size_t second) { return first < second; });//Ascending order

        if (!actCorrsImg1TNFromLast_Idx.empty())//Adapt the indices for TN (single keypoints without a match)
        {
            adaptIndicesNoDel(actCorrsImg1TNFromLast_Idx, delList3D);
        }
        if (!actCorrsImg2TNFromLast_Idx.empty())//Adapt the indices for TN (single keypoints without a match)
        {
            adaptIndicesNoDel(actCorrsImg2TNFromLast_Idx, delList3D);
        }
        adaptIndicesNoDel(actCorrsImg12TPFromLast_Idx, delList3D);
        deleteVecEntriesbyIdx(actImgPointCloudFromLast, delList3D);
        deleteVecEntriesbyIdx(actCorrsImg12TPFromLast_IdxWorld, delList3D);

//        sort(delListCorrs.begin(), delListCorrs.end(), [](size_t first, size_t second) { return first < second; });

        deleteVecEntriesbyIdx(actCorrsImg12TPFromLast_Idx, delListCorrs);
        deleteMatEntriesByIdx(actCorrsImg1TPFromLast, delListCorrs, false);
        deleteMatEntriesByIdx(actCorrsImg2TPFromLast, delListCorrs, false);
//        if(!checkCorr3DConsistency()){
//            throw SequenceException("Correspondences are not projections of 3D points!");
//        }
    }

    return actDelNr;
}

//Initialize region ROIs and masks
void genStereoSequ::genRegMasks() {
    //Construct valid areas for every region
    regmasks = vector<vector<Mat>>(3, vector<Mat>(3));
    regmasksROIs = vector<vector<cv::Rect>>(3, vector<cv::Rect>(3));
    regROIs = vector<vector<cv::Rect>>(3, vector<cv::Rect>(3));
    Size imgSi13 = Size(imgSize.width / 3, imgSize.height / 3);
    Mat validRect = Mat::ones(imgSize, CV_8UC1);
    const float overSi = 1.25f;//Allows the expension of created areas outside its region by a given percentage

    for (int y = 0; y < 3; y++) {
        cv::Point2i pl1, pr1, pl2, pr2;
        pl1.y = y * imgSi13.height;
        pl2.y = pl1.y;
        if (y < 2) {
            pr1.y = pl1.y + (int) (overSi * (float) imgSi13.height);
            pr2.y = pl2.y + imgSi13.height;
        } else {
            pr1.y = imgSize.height;
            pr2.y = imgSize.height;
        }
        if (y > 0) {
            pl1.y -= (int) ((overSi - 1.f) * (float) imgSi13.height);
        }
        for (int x = 0; x < 3; x++) {
            pl1.x = x * imgSi13.width;
            pl2.x = pl1.x;
            if (x < 2) {
                pr1.x = pl1.x + (int) (overSi * (float) imgSi13.width);
                pr2.x = pl2.x + imgSi13.width;
            } else {
                pr1.x = imgSize.width;
                pr2.x = imgSize.width;
            }
            if (x > 0) {
                pl1.x -= (int) ((overSi - 1.f) * (float) imgSi13.width);
            }
            Rect vROI = Rect(pl1, pr1);
            regmasksROIs[y][x] = vROI;
            regmasks[y][x] = Mat::zeros(imgSize, CV_8UC1);
            validRect(vROI).copyTo(regmasks[y][x](vROI));
            Rect vROIo = Rect(pl2, pr2);
            regROIs[y][x] = vROIo;
        }
    }
}

//Generates a depth map with the size of the image where each pixel value corresponds to the depth
void genStereoSequ::genDepthMaps() {
    int minSi = (int) sqrt(minDArea);

    int maskEnlarge = 0;
    for (size_t y = 0; y < 3; y++) {
        for (size_t x = 0; x < 3; x++) {
            int32_t tmp;
            if (!seedsNearNNDist[y][x].empty()) {
                tmp = *std::max_element(seedsNearNNDist[y][x].begin(), seedsNearNNDist[y][x].end());
                if (tmp > maskEnlarge) {
                    maskEnlarge = tmp;
                }
            }
            if (!seedsMidNNDist[y][x].empty()) {
                tmp = *std::max_element(seedsMidNNDist[y][x].begin(), seedsMidNNDist[y][x].end());
                if (tmp > maskEnlarge) {
                    maskEnlarge = tmp;
                }
            }
            if (!seedsFarNNDist[y][x].empty()) {
                tmp = *std::max_element(seedsFarNNDist[y][x].begin(), seedsFarNNDist[y][x].end());
                if (tmp > maskEnlarge) {
                    maskEnlarge = tmp;
                }
            }
        }
    }

    cv::Mat noGenMaskB = Mat::zeros(imgSize.height + 2 * maskEnlarge, imgSize.width + 2 * maskEnlarge, CV_8UC1);
    Mat noGenMaskB2 = noGenMaskB.clone();
    cv::Mat noGenMask = noGenMaskB(Range(maskEnlarge, imgSize.height + maskEnlarge),
                                   Range(maskEnlarge, imgSize.width + maskEnlarge));
    Mat noGenMask2 = noGenMaskB2(Range(maskEnlarge, imgSize.height + maskEnlarge),
                                 Range(maskEnlarge, imgSize.width + maskEnlarge));

    //Get an ordering of the different depth area sizes for every region
    cv::Mat beginDepth = cv::Mat(3, 3, CV_32SC3);
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            int32_t maxAPReg[3];
            maxAPReg[0] = areaPRegNear[actCorrsPRIdx].at<int32_t>(y, x);
            maxAPReg[1] = areaPRegMid[actCorrsPRIdx].at<int32_t>(y, x);
            maxAPReg[2] = areaPRegFar[actCorrsPRIdx].at<int32_t>(y, x);
            std::ptrdiff_t pdiff = min_element(maxAPReg, maxAPReg + 3) - maxAPReg;
            beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[0] = (int32_t)pdiff;
            if (maxAPReg[(pdiff + 1) % 3] < maxAPReg[(pdiff + 2) % 3]) {
                beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[1] = ((int32_t)pdiff + 1) % 3;
                beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2] = ((int32_t)pdiff + 2) % 3;
            } else {
                beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2] = ((int32_t)pdiff + 1) % 3;
                beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[1] = ((int32_t)pdiff + 2) % 3;
            }
        }
    }

    //Get the average area for every seed position
    Mat meanNearA = Mat::zeros(3, 3, CV_32SC2);
    Mat meanMidA = Mat::zeros(3, 3, CV_32SC2);
    Mat meanFarA = Mat::zeros(3, 3, CV_32SC2);
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            int32_t checkArea = 0;
            if (!seedsNear[y][x].empty()) {
                meanNearA.at<cv::Vec<int32_t, 2>>(y, x)[0] =
                        areaPRegNear[actCorrsPRIdx].at<int32_t>(y, x) / (int32_t) seedsNear[y][x].size();
                meanNearA.at<cv::Vec<int32_t, 2>>(y, x)[1] = (int32_t) sqrt(
                        (double) meanNearA.at<cv::Vec<int32_t, 2>>(y, x)[0] / M_PI);
            }
            checkArea += areaPRegNear[actCorrsPRIdx].at<int32_t>(y, x);
            if (!seedsMid[y][x].empty()) {
                meanMidA.at<cv::Vec<int32_t, 2>>(y, x)[0] =
                        areaPRegMid[actCorrsPRIdx].at<int32_t>(y, x) / (int32_t) seedsMid[y][x].size();
                meanMidA.at<cv::Vec<int32_t, 2>>(y, x)[1] = (int32_t) sqrt(
                        (double) meanMidA.at<cv::Vec<int32_t, 2>>(y, x)[0] / M_PI);
            }
            checkArea += areaPRegMid[actCorrsPRIdx].at<int32_t>(y, x);
            if (!seedsFar[y][x].empty()) {
                meanFarA.at<cv::Vec<int32_t, 2>>(y, x)[0] =
                        areaPRegFar[actCorrsPRIdx].at<int32_t>(y, x) / (int32_t) seedsFar[y][x].size();
                meanFarA.at<cv::Vec<int32_t, 2>>(y, x)[1] = (int32_t) sqrt(
                        (double) meanFarA.at<cv::Vec<int32_t, 2>>(y, x)[0] / M_PI);
            }
            checkArea += areaPRegFar[actCorrsPRIdx].at<int32_t>(y, x);
            if(verbose & PRINT_WARNING_MESSAGES) {
                if (checkArea != regROIs[y][x].area()) {
                    cout << "Sum of static depth areas (" << checkArea << ") does not correspond to area of region ("
                         << regROIs[y][x].area() << ")!" << endl;
                }
            }
        }
    }

    //Reserve a little bit of space for depth areas generated later on (as they are larger)
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            for (int i = 2; i >= 1; i--) {
                switch (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[i]) {
                    case 0:
                        for (size_t j = 0; j < seedsNear[y][x].size(); j++) {
                            cv::Point3_<int32_t> pt = seedsNear[y][x][j];
                            Mat part, rmask;
                            int32_t nnd = seedsNearNNDist[y][x][j];
                            int32_t useRad = max(min(max((nnd - 1) / 2, meanNearA.at<cv::Vec<int32_t, 2>>(y, x)[1]),
                                                 nnd - 2),1);
                            int32_t offset = maskEnlarge - useRad;
                            int32_t offset2 = maskEnlarge + useRad;
                            getRandMask(rmask, meanNearA.at<cv::Vec<int32_t, 2>>(y, x)[0], useRad, minSi);
                            if (i == 2) {
                                part = noGenMaskB2(Range(pt.y + offset, pt.y + offset2),
                                                   Range(pt.x + offset, pt.x + offset2));
                            } else {
                                part = noGenMaskB(Range(pt.y + offset, pt.y + offset2),
                                                  Range(pt.x + offset, pt.x + offset2));
                            }
                            part |= rmask;
                        }
                        break;
                    case 1:
                        for (size_t j = 0; j < seedsMid[y][x].size(); j++) {
                            cv::Point3_<int32_t> pt = seedsMid[y][x][j];
                            Mat part, rmask;
                            int32_t nnd = seedsMidNNDist[y][x][j];
                            int32_t useRad = max(min(max((nnd - 1) / 2, meanMidA.at<cv::Vec<int32_t, 2>>(y, x)[1]),
                                                 nnd - 2),1);
                            int32_t offset = maskEnlarge - useRad;
                            int32_t offset2 = maskEnlarge + useRad;
                            getRandMask(rmask, meanMidA.at<cv::Vec<int32_t, 2>>(y, x)[0], useRad, minSi);
                            if (i == 2) {
                                part = noGenMaskB2(Range(pt.y + offset, pt.y + offset2),
                                                   Range(pt.x + offset, pt.x + offset2));
                            } else {
                                part = noGenMaskB(Range(pt.y + offset, pt.y + offset2),
                                                  Range(pt.x + offset, pt.x + offset2));
                            }
                            part |= rmask;
                        }
                        break;
                    case 2:
                        for (size_t j = 0; j < seedsFar[y][x].size(); j++) {
                            cv::Point3_<int32_t> pt = seedsFar[y][x][j];
                            Mat part, rmask;
                            int32_t nnd = seedsFarNNDist[y][x][j];
                            int32_t useRad = max(min(max((nnd - 1) / 2, meanFarA.at<cv::Vec<int32_t, 2>>(y, x)[1]),
                                                 nnd - 2),1);
                            int32_t offset = maskEnlarge - useRad;
                            int32_t offset2 = maskEnlarge + useRad;
                            getRandMask(rmask, meanFarA.at<cv::Vec<int32_t, 2>>(y, x)[0], useRad, minSi);
                            if (i == 2) {
                                part = noGenMaskB2(Range(pt.y + offset, pt.y + offset2),
                                                   Range(pt.x + offset, pt.x + offset2));
                            } else {
                                part = noGenMaskB(Range(pt.y + offset, pt.y + offset2),
                                                  Range(pt.x + offset, pt.x + offset2));
                            }
                            part |= rmask;
                        }
                        break;
                    default:
                        break;
                }
            }
        }
    }
    noGenMaskB |= noGenMaskB2;

    //Show the masks
    if (verbose & SHOW_BUILD_PROC_STATIC_OBJ) {
        if(!writeIntermediateImg(noGenMask, "static_obj_mask_largest_2_depths") ||
                !writeIntermediateImg(noGenMask2, "static_obj_mask_largest_depth")) {
            namedWindow("Mask for largest 2 depths", WINDOW_AUTOSIZE);
            imshow("Mask for largest 2 depths", noGenMask);
            namedWindow("Mask for largest depth", WINDOW_AUTOSIZE);
            imshow("Mask for largest depth", noGenMask2);
            waitKey(0);
            destroyWindow("Mask for largest 2 depths");
            destroyWindow("Mask for largest depth");
        }
    }

    //Show the region masks
    /*namedWindow("Region mask", WINDOW_AUTOSIZE);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            imshow("Region mask", (regmasks[y][x] > 0));
            waitKey(0);
        }
    }
    destroyWindow("Region mask");*/

    //Create first layer of depth areas
    std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> actPosSeedsNear(3,
                                                                               std::vector<std::vector<cv::Point_<int32_t>>>(
                                                                                       3));
    std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> actPosSeedsMid(3,
                                                                              std::vector<std::vector<cv::Point_<int32_t>>>(
                                                                                      3));
    std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> actPosSeedsFar(3,
                                                                              std::vector<std::vector<cv::Point_<int32_t>>>(
                                                                                      3));
    std::vector<std::vector<std::vector<size_t>>> nrIterPerSeedNear(3, std::vector<std::vector<size_t>>(3));
    std::vector<std::vector<std::vector<size_t>>> nrIterPerSeedMid(3, std::vector<std::vector<size_t>>(3));
    std::vector<std::vector<std::vector<size_t>>> nrIterPerSeedFar(3, std::vector<std::vector<size_t>>(3));
    std::vector<std::vector<int32_t>> actAreaNear(3, vector<int32_t>(3, 0));
    std::vector<std::vector<int32_t>> actAreaMid(3, vector<int32_t>(3, 0));
    std::vector<std::vector<int32_t>> actAreaFar(3, vector<int32_t>(3, 0));
    std::vector<std::vector<unsigned char>> dilateOpNear(3, vector<unsigned char>(3, 0));
    std::vector<std::vector<unsigned char>> dilateOpMid(3, vector<unsigned char>(3, 0));
    std::vector<std::vector<unsigned char>> dilateOpFar(3, vector<unsigned char>(3, 0));
    depthAreaMap = Mat::zeros(imgSize, CV_8UC1);
    Mat actUsedAreaNear = Mat::zeros(imgSize, CV_8UC1);
    Mat actUsedAreaMid = Mat::zeros(imgSize, CV_8UC1);
    Mat actUsedAreaFar = Mat::zeros(imgSize, CV_8UC1);
    Mat neighborRegMask = Mat::zeros(imgSize, CV_8UC1);
    //Init actual positions
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (!seedsNear[y][x].empty() && (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2] != 0)) {
                actPosSeedsNear[y][x].resize(seedsNear[y][x].size());
                nrIterPerSeedNear[y][x].resize(seedsNear[y][x].size(), 0);
                for (size_t i = 0; i < seedsNear[y][x].size(); i++) {
                    int ix = seedsNear[y][x][i].x;
                    int iy = seedsNear[y][x][i].y;
                    actPosSeedsNear[y][x][i].x = ix;
                    actPosSeedsNear[y][x][i].y = iy;
                    depthAreaMap.at<unsigned char>(iy, ix) = 1;
                    actUsedAreaNear.at<unsigned char>(iy, ix) = 1;
                    neighborRegMask.at<unsigned char>(iy, ix) = (unsigned char) (y * 3 + x + 1);
                    actAreaNear[y][x]++;
                }
            }
            if (!seedsMid[y][x].empty() && (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2] != 1)) {
                actPosSeedsMid[y][x].resize(seedsMid[y][x].size());
                nrIterPerSeedMid[y][x].resize(seedsMid[y][x].size(), 0);
                for (size_t i = 0; i < seedsMid[y][x].size(); i++) {
                    int ix = seedsMid[y][x][i].x;
                    int iy = seedsMid[y][x][i].y;
                    actPosSeedsMid[y][x][i].x = ix;
                    actPosSeedsMid[y][x][i].y = iy;
                    depthAreaMap.at<unsigned char>(iy, ix) = 2;
                    actUsedAreaMid.at<unsigned char>(iy, ix) = 1;
                    neighborRegMask.at<unsigned char>(iy, ix) = (unsigned char) (y * 3 + x + 1);
                    actAreaMid[y][x]++;
                }
            }
            if (!seedsFar[y][x].empty() && (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2] != 2)) {
                actPosSeedsFar[y][x].resize(seedsFar[y][x].size());
                nrIterPerSeedFar[y][x].resize(seedsFar[y][x].size(), 0);
                for (size_t i = 0; i < seedsFar[y][x].size(); i++) {
                    int ix = seedsFar[y][x][i].x;
                    int iy = seedsFar[y][x][i].y;
                    actPosSeedsFar[y][x][i].x = ix;
                    actPosSeedsFar[y][x][i].y = iy;
                    depthAreaMap.at<unsigned char>(iy, ix) = 3;
                    actUsedAreaFar.at<unsigned char>(iy, ix) = 1;
                    neighborRegMask.at<unsigned char>(iy, ix) = (unsigned char) (y * 3 + x + 1);
                    actAreaFar[y][x]++;
                }
            }
        }
    }

    //Create depth areas beginning with the smallest areas (near, mid, or far) per region
    //Also create depth areas for the second smallest areas
    Size imgSiM1 = Size(imgSize.width - 1, imgSize.height - 1);
    size_t visualizeMask = 0;
    for (int j = 0; j < 2; j++) {
        if (j > 0) {
            noGenMask = noGenMask2;
        }
        bool areasNFinish[3][3] = {{true, true, true}, {true, true, true}, {true, true, true}};
        while (areasNFinish[0][0] || areasNFinish[0][1] || areasNFinish[0][2] ||
               areasNFinish[1][0] || areasNFinish[1][1] || areasNFinish[1][2] ||
               areasNFinish[2][0] || areasNFinish[2][1] || areasNFinish[2][2]) {
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    if (!areasNFinish[y][x]) continue;
                    switch (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[j]) {
                        case 0:
                            if (!actPosSeedsNear[y][x].empty()) {
                                for (size_t i = 0; i < actPosSeedsNear[y][x].size(); i++) {
                                    if (areasNFinish[y][x]) {

                                        /*Mat beforeAdding =  actUsedAreaNear(regmasksROIs[y][x]) & (neighborRegMask(regmasksROIs[y][x]) == (unsigned char) (y * 3 + x));
                                        int32_t Asv = actAreaNear[y][x];*/

                                        areasNFinish[y][x] = addAdditionalDepth(1,
                                                                                depthAreaMap,
                                                                                actUsedAreaNear,
                                                                                noGenMask,
                                                                                regmasks[y][x],
                                                                                actPosSeedsNear[y][x][i],
                                                                                actPosSeedsNear[y][x][i],
                                                                                actAreaNear[y][x],
                                                                                areaPRegNear[actCorrsPRIdx].at<int32_t>(
                                                                                        y,
                                                                                        x),
                                                                                imgSiM1,
                                                                                cv::Point_<int32_t>(
                                                                                        seedsNear[y][x][i].x,
                                                                                        seedsNear[y][x][i].y),
                                                                                regmasksROIs[y][x],
                                                                                nrIterPerSeedNear[y][x][i],
                                                                                dilateOpNear[y][x],
                                                                                neighborRegMask,
                                                                                (unsigned char) (y * 3 + x + 1));

                                        /*Mat afterAdding =  actUsedAreaNear(regmasksROIs[y][x]) & (neighborRegMask(regmasksROIs[y][x]) == (unsigned char) (y * 3 + x));
                                        int realAreaBeforeDil = cv::countNonZero(afterAdding);
                                        if(realAreaBeforeDil != actAreaNear[y][x])
                                        {
                                            cout << "Area difference: " << realAreaBeforeDil - actAreaNear[y][x] << endl;
                                            cout << "Area diff between last and actual values: " << actAreaNear[y][x] - Asv << endl;
                                            Mat addingDiff = afterAdding ^ beforeAdding;
                                            namedWindow("Before", WINDOW_AUTOSIZE);
                                            namedWindow("After", WINDOW_AUTOSIZE);
                                            namedWindow("Diff", WINDOW_AUTOSIZE);
                                            namedWindow("Mask", WINDOW_AUTOSIZE);
                                            namedWindow("All Regions", WINDOW_AUTOSIZE);
                                            namedWindow("Neighbours", WINDOW_AUTOSIZE);
                                            imshow("Before", (beforeAdding > 0));
                                            imshow("After", (afterAdding > 0));
                                            imshow("Diff", (addingDiff > 0));
                                            imshow("Mask", noGenMask(regmasksROIs[y][x]));
                                            Mat colorMapImg;
                                            unsigned char clmul = 255 / 3;
                                            // Apply the colormap:
                                            applyColorMap(depthAreaMap(regmasksROIs[y][x]) * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
                                            imshow("All Regions", colorMapImg);
                                            clmul = 255 / 9;
                                            applyColorMap(neighborRegMask(regmasksROIs[y][x]) * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
                                            imshow("Neighbours", colorMapImg);
                                            waitKey(0);
                                            destroyWindow("Before");
                                            destroyWindow("After");
                                            destroyWindow("Diff");
                                            destroyWindow("Mask");
                                            destroyWindow("All Regions");
                                            destroyWindow("Neighbours");
                                        }*/
                                    } else {
                                        break;
                                    }
                                }
                            } else
                                areasNFinish[y][x] = false;
                            break;
                        case 1:
                            if (!actPosSeedsMid[y][x].empty()) {
                                for (size_t i = 0; i < actPosSeedsMid[y][x].size(); i++) {
                                    if (areasNFinish[y][x]) {

                                        /*Mat beforeAdding =  actUsedAreaMid(regmasksROIs[y][x]) & (neighborRegMask(regmasksROIs[y][x]) == (unsigned char) (y * 3 + x));
                                        int32_t Asv = actAreaMid[y][x];*/

                                        areasNFinish[y][x] = addAdditionalDepth(2,
                                                                                depthAreaMap,
                                                                                actUsedAreaMid,
                                                                                noGenMask,
                                                                                regmasks[y][x],
                                                                                actPosSeedsMid[y][x][i],
                                                                                actPosSeedsMid[y][x][i],
                                                                                actAreaMid[y][x],
                                                                                areaPRegMid[actCorrsPRIdx].at<int32_t>(
                                                                                        y,
                                                                                        x),
                                                                                imgSiM1,
                                                                                cv::Point_<int32_t>(seedsMid[y][x][i].x,
                                                                                                    seedsMid[y][x][i].y),
                                                                                regmasksROIs[y][x],
                                                                                nrIterPerSeedMid[y][x][i],
                                                                                dilateOpMid[y][x],
                                                                                neighborRegMask,
                                                                                (unsigned char) (y * 3 + x + 1));

                                        /*Mat afterAdding =  actUsedAreaMid(regmasksROIs[y][x]) & (neighborRegMask(regmasksROIs[y][x]) == (unsigned char) (y * 3 + x));
                                        int realAreaBeforeDil = cv::countNonZero(afterAdding);
                                        if(realAreaBeforeDil != actAreaMid[y][x])
                                        {
                                            cout << "Area difference: " << realAreaBeforeDil - actAreaMid[y][x] << endl;
                                            cout << "Area diff between last and actual values: " << actAreaMid[y][x] - Asv << endl;
                                            Mat addingDiff = afterAdding ^ beforeAdding;
                                            namedWindow("Before", WINDOW_AUTOSIZE);
                                            namedWindow("After", WINDOW_AUTOSIZE);
                                            namedWindow("Diff", WINDOW_AUTOSIZE);
                                            namedWindow("Mask", WINDOW_AUTOSIZE);
                                            namedWindow("All Regions", WINDOW_AUTOSIZE);
                                            namedWindow("Neighbours", WINDOW_AUTOSIZE);
                                            imshow("Before", (beforeAdding > 0));
                                            imshow("After", (afterAdding > 0));
                                            imshow("Diff", (addingDiff > 0));
                                            imshow("Mask", noGenMask(regmasksROIs[y][x]));
                                            Mat colorMapImg;
                                            unsigned char clmul = 255 / 3;
                                            // Apply the colormap:
                                            applyColorMap(depthAreaMap(regmasksROIs[y][x]) * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
                                            imshow("All Regions", colorMapImg);
                                            clmul = 255 / 9;
                                            applyColorMap(neighborRegMask(regmasksROIs[y][x]) * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
                                            imshow("Neighbours", colorMapImg);
                                            waitKey(0);
                                            destroyWindow("Before");
                                            destroyWindow("After");
                                            destroyWindow("Diff");
                                            destroyWindow("Mask");
                                            destroyWindow("All Regions");
                                            destroyWindow("Neighbours");
                                        }*/
                                    } else {
                                        break;
                                    }
                                }
                            } else
                                areasNFinish[y][x] = false;
                            break;
                        case 2:
                            if (!actPosSeedsFar[y][x].empty()) {
                                for (size_t i = 0; i < actPosSeedsFar[y][x].size(); i++) {
                                    if (areasNFinish[y][x]) {

                                        /*Mat beforeAdding =  actUsedAreaFar(regmasksROIs[y][x]) & (neighborRegMask(regmasksROIs[y][x]) == (unsigned char) (y * 3 + x));
                                        int32_t Asv = actAreaFar[y][x];*/

                                        areasNFinish[y][x] = addAdditionalDepth(3,
                                                                                depthAreaMap,
                                                                                actUsedAreaFar,
                                                                                noGenMask,
                                                                                regmasks[y][x],
                                                                                actPosSeedsFar[y][x][i],
                                                                                actPosSeedsFar[y][x][i],
                                                                                actAreaFar[y][x],
                                                                                areaPRegFar[actCorrsPRIdx].at<int32_t>(
                                                                                        y,
                                                                                        x),
                                                                                imgSiM1,
                                                                                cv::Point_<int32_t>(seedsFar[y][x][i].x,
                                                                                                    seedsFar[y][x][i].y),
                                                                                regmasksROIs[y][x],
                                                                                nrIterPerSeedFar[y][x][i],
                                                                                dilateOpFar[y][x],
                                                                                neighborRegMask,
                                                                                (unsigned char) (y * 3 + x + 1));

                                        /*Mat afterAdding =  actUsedAreaFar(regmasksROIs[y][x]) & (neighborRegMask(regmasksROIs[y][x]) == (unsigned char) (y * 3 + x));
                                        int realAreaBeforeDil = cv::countNonZero(afterAdding);
                                        if(realAreaBeforeDil != actAreaFar[y][x])
                                        {
                                            cout << "Area difference: " << realAreaBeforeDil - actAreaFar[y][x] << endl;
                                            cout << "Area diff between last and actual values: " << actAreaFar[y][x] - Asv << endl;
                                            Mat addingDiff = afterAdding ^ beforeAdding;
                                            namedWindow("Before", WINDOW_AUTOSIZE);
                                            namedWindow("After", WINDOW_AUTOSIZE);
                                            namedWindow("Diff", WINDOW_AUTOSIZE);
                                            namedWindow("Mask", WINDOW_AUTOSIZE);
                                            namedWindow("All Regions", WINDOW_AUTOSIZE);
                                            namedWindow("Neighbours", WINDOW_AUTOSIZE);
                                            imshow("Before", (beforeAdding > 0));
                                            imshow("After", (afterAdding > 0));
                                            imshow("Diff", (addingDiff > 0));
                                            imshow("Mask", noGenMask(regmasksROIs[y][x]));
                                            Mat colorMapImg;
                                            unsigned char clmul = 255 / 3;
                                            // Apply the colormap:
                                            applyColorMap(depthAreaMap(regmasksROIs[y][x]) * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
                                            imshow("All Regions", colorMapImg);
                                            clmul = 255 / 9;
                                            applyColorMap(neighborRegMask(regmasksROIs[y][x]) * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
                                            imshow("Neighbours", colorMapImg);
                                            waitKey(0);
                                            destroyWindow("Before");
                                            destroyWindow("After");
                                            destroyWindow("Diff");
                                            destroyWindow("Mask");
                                            destroyWindow("All Regions");
                                            destroyWindow("Neighbours");
                                        }*/
                                    } else {
                                        break;
                                    }
                                }
                            } else
                                areasNFinish[y][x] = false;
                            break;
                        default:
                            break;
                    }
                }
            }
            if (verbose & SHOW_BUILD_PROC_STATIC_OBJ) {
                if (visualizeMask % 200 == 0) {
                    Mat colorMapImg;
                    unsigned char clmul = 255 / 3;
                    // Apply the colormap:
                    applyColorMap(depthAreaMap * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
                    if(!writeIntermediateImg(colorMapImg, "static_obj_depth_areas_creation_process_step_" + std::to_string(visualizeMask))){
                        namedWindow("Static object depth areas creation process", WINDOW_AUTOSIZE);
                        imshow("Static object depth areas creation process", colorMapImg);
                        waitKey(0);
                        destroyWindow("Static object depth areas creation process");
                    }
                }
                visualizeMask++;
            }
        }
    }

    //Show the intermediate result
    if (verbose & SHOW_BUILD_PROC_STATIC_OBJ) {
        unsigned char clmul = 255 / 3;
        // Apply the colormap:
        Mat colorMapImg;
        applyColorMap(depthAreaMap * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
        if(!writeIntermediateImg(colorMapImg, "static_obj_depth_areas_after_filling_2_depths_per_region")){
            namedWindow("Static object depth areas after filling 2 depths per region", WINDOW_AUTOSIZE);
            imshow("Static object depth areas after filling 2 depths per region", colorMapImg);
            waitKey(0);
            destroyWindow("Static object depth areas after filling 2 depths per region");
        }
    }

    /*//Show the mask
    {
        namedWindow("Mask for largest depth", WINDOW_AUTOSIZE);
        imshow("Mask for largest depth", noGenMask2);
        waitKey(0);
        destroyWindow("Mask for largest depth");
    }*/

    //Fill the remaining areas:
    //Generate the (largest) depth areas per region independent of the largest & different depth areas of other regions
    Mat maskNear = Mat::zeros(imgSize, CV_8UC1);
    Mat maskMid = Mat::zeros(imgSize, CV_8UC1);
    Mat maskFar = Mat::zeros(imgSize, CV_8UC1);
    int32_t fillAreas[3] = {0, 0, 0};
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            switch (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2]) {
                case 0:
                    maskNear(regROIs[y][x]) |= noGenMask2(regROIs[y][x]);
                    fillAreas[0] += areaPRegNear[actCorrsPRIdx].at<int32_t>(y, x);
                    break;
                case 1:
                    maskMid(regROIs[y][x]) |= noGenMask2(regROIs[y][x]);
                    fillAreas[1] += areaPRegMid[actCorrsPRIdx].at<int32_t>(y, x);
                    break;
                case 2:
                    maskFar(regROIs[y][x]) |= noGenMask2(regROIs[y][x]);
                    fillAreas[2] += areaPRegFar[actCorrsPRIdx].at<int32_t>(y, x);
                    break;
                default:
                    break;
            }
        }
    }
    int32_t actualAreas[3] = {0, 0, 0};
    actualAreas[0] = cv::countNonZero(maskNear);
    actualAreas[1] = cv::countNonZero(maskMid);
    actualAreas[2] = cv::countNonZero(maskFar);
    fillRemainingAreas(maskNear, depthAreaMap, fillAreas[0], actualAreas[0]);
    fillRemainingAreas(maskMid, depthAreaMap, fillAreas[1], actualAreas[1]);
    fillRemainingAreas(maskFar, depthAreaMap, fillAreas[2], actualAreas[2]);

    //Show the masks
    /*{
        Mat colorMapImg;
        unsigned char clmul = 255 / 3;
        // Apply the colormap:
        Mat completeDepthMap = (maskNear & Mat::ones(imgSize, CV_8UC1)) * clmul;
        completeDepthMap |= (maskMid & Mat::ones(imgSize, CV_8UC1)) * (clmul * 2);
        completeDepthMap |= maskFar;
        applyColorMap(completeDepthMap, colorMapImg, cv::COLORMAP_RAINBOW);
        namedWindow("Static object depth areas", WINDOW_AUTOSIZE);
        imshow("Static object depth areas", colorMapImg);
        waitKey(0);
        destroyWindow("Static object depth areas");
    }*/

    //Get overlaps of filled areas (3 different) and remove them
    Mat overlap3 = maskNear & maskMid & maskFar;
    int nr_overlap3 = cv::countNonZero(overlap3);
    if (nr_overlap3) {
        Mat overlap3sv = overlap3.clone();
        int overlapDel = nr_overlap3 / 3;
        //Remove small mid and far areas (only near areas remain in overlap areas)
        removeNrFilledPixels(cv::Size(3, 3), imgSize, overlap3, overlapDel);
        Mat changeMask = ((overlap3 ^ overlap3sv) == 0);
        maskMid &= changeMask;
        maskFar &= changeMask;
        overlap3sv = overlap3.clone();
        //Remove small near and far areas (only mid areas remain in overlap areas)
        removeNrFilledPixels(cv::Size(3, 3), imgSize, overlap3, overlapDel);
        changeMask = ((overlap3 ^ overlap3sv) == 0);
        maskNear &= changeMask;
        maskFar &= changeMask;
        //Remove small near and mid areas (only far areas remain in overlap areas)
        changeMask = (overlap3 == 0);
        maskNear &= changeMask;
        maskMid &= changeMask;
    }
    //Get overlaps of filled areas (2 different) and remove them
    delOverlaps2(maskNear, maskMid);
    delOverlaps2(maskNear, maskFar);
    delOverlaps2(maskFar, maskMid);

    //Show the masks
    if (verbose & SHOW_BUILD_PROC_STATIC_OBJ) {
        Mat colorMapImg;
        unsigned char clmul = 255 / 3;
        // Apply the colormap:
        Mat completeDepthMap = (maskNear & Mat::ones(imgSize, CV_8UC1)) * clmul;
        completeDepthMap |= (maskMid & Mat::ones(imgSize, CV_8UC1)) * (clmul * 2);
        completeDepthMap |= maskFar;
        applyColorMap(completeDepthMap, colorMapImg, cv::COLORMAP_RAINBOW);
        if(!writeIntermediateImg(colorMapImg, "largest_static_object_depth_areas_before_final_dilation")){
            namedWindow("Largest static object depth areas before final dilation", WINDOW_AUTOSIZE);
            imshow("Largest static object depth areas before final dilation", colorMapImg);
            waitKey(0);
            destroyWindow("Largest static object depth areas before final dilation");
        }
    }

    //Try to fill the remaining gaps using dilation
    const int maxCnt = 20;
    int cnt = 0;
    bool nFinished[3] = {true, true, true};
    actualAreas[0] = cv::countNonZero(maskNear);
    actualAreas[1] = cv::countNonZero(maskMid);
    actualAreas[2] = cv::countNonZero(maskFar);
    if (actualAreas[0] >= fillAreas[0]) {
        nFinished[0] = false;
    }
    if (actualAreas[1] >= fillAreas[1]) {
        nFinished[1] = false;
    }
    if (actualAreas[2] >= fillAreas[2]) {
        nFinished[2] = false;
    }
    while ((nFinished[0] || nFinished[1] || nFinished[2]) && (cnt < maxCnt)) {
        if (nFinished[0]) {
            if (!fillRemainingAreas(maskNear, depthAreaMap, fillAreas[0], actualAreas[0], maskMid, maskFar)) {
                nFinished[0] = false;
            }
            if (actualAreas[0] >= fillAreas[0]) {
                nFinished[0] = false;
            }
        }
        if (nFinished[1]) {
            if (fillRemainingAreas(maskMid, depthAreaMap, fillAreas[1], actualAreas[1], maskNear, maskFar)) {
                nFinished[1] = false;
            }
            if (actualAreas[1] >= fillAreas[1]) {
                nFinished[1] = false;
            }
        }
        if (nFinished[2]) {
            if (fillRemainingAreas(maskFar, depthAreaMap, fillAreas[2], actualAreas[2], maskNear, maskMid)) {
                nFinished[2] = false;
            }
            if (actualAreas[2] >= fillAreas[2]) {
                nFinished[2] = false;
            }
        }
        cnt++;
    }
    if (actualAreas[0] < fillAreas[0]) {
        nFinished[0] = true;
    }
    if (actualAreas[1] < fillAreas[1]) {
        nFinished[1] = true;
    }
    if (actualAreas[2] < fillAreas[2]) {
        nFinished[2] = true;
    }

    //Show the masks
    if (verbose & SHOW_BUILD_PROC_STATIC_OBJ) {
        Mat colorMapImg;
        unsigned char clmul = 255 / 3;
        // Apply the colormap:
        Mat completeDepthMap = (maskNear & Mat::ones(imgSize, CV_8UC1)) * clmul;
        completeDepthMap |= (maskMid & Mat::ones(imgSize, CV_8UC1)) * (clmul * 2);
        completeDepthMap |= maskFar;
        applyColorMap(completeDepthMap, colorMapImg, cv::COLORMAP_RAINBOW);
        if(!writeIntermediateImg(colorMapImg, "static_object_depth_largest_areas")){
            namedWindow("Static object depth areas (largest areas)", WINDOW_AUTOSIZE);
            imshow("Static object depth areas (largest areas)", colorMapImg);
            waitKey(0);
            destroyWindow("Static object depth areas (largest areas)");
        }
    }

    //Combine created masks
    Mat maskNMF1s = maskNear & Mat::ones(imgSize, actUsedAreaNear.type());
    actUsedAreaNear |= maskNMF1s;
    depthAreaMap |= maskNMF1s;
    maskNMF1s = maskMid & Mat::ones(imgSize, actUsedAreaNear.type());
    actUsedAreaMid |= maskNMF1s;
    maskNMF1s *= 2;
    depthAreaMap |= maskNMF1s;
    maskNMF1s = maskFar & Mat::ones(imgSize, actUsedAreaNear.type());
    actUsedAreaFar |= maskNMF1s;
    maskNMF1s *= 3;
    depthAreaMap |= maskNMF1s;

    //Show the result
    if (verbose & SHOW_BUILD_PROC_STATIC_OBJ) {
        unsigned char clmul = 255 / 3;
        // Apply the colormap:
        Mat colorMapImg;
        applyColorMap(depthAreaMap * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
        if(!writeIntermediateImg(colorMapImg, "static_object_depth_areas_before_glob_area_fill")){
            namedWindow("Static object depth areas before glob area fill", WINDOW_AUTOSIZE);
            imshow("Static object depth areas before glob area fill", colorMapImg);
            waitKey(0);
            destroyWindow("Static object depth areas before glob area fill");
        }
    }

    //Fill the remaining areas
//    if(nFinished[0] || nFinished[1] || nFinished[2]) {
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            Mat fillMask =
                    (depthAreaMap(regROIs[y][x]) == 0) &
                    Mat::ones(regROIs[y][x].height, regROIs[y][x].width, CV_8UC1);
            switch (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2]) {
                case 0:
                    actUsedAreaNear(regROIs[y][x]) |= fillMask;
                    depthAreaMap(regROIs[y][x]) |= fillMask;
                    break;
                case 1:
                    actUsedAreaMid(regROIs[y][x]) |= fillMask;
                    fillMask *= 2;
                    depthAreaMap(regROIs[y][x]) |= fillMask;
                    break;
                case 2:
                    actUsedAreaFar(regROIs[y][x]) |= fillMask;
                    fillMask *= 3;
                    depthAreaMap(regROIs[y][x]) |= fillMask;
                    break;
                default:
                    break;
            }
        }
    }
//    }

    //Show the result
    if (verbose & (SHOW_BUILD_PROC_STATIC_OBJ | SHOW_STATIC_OBJ_DISTANCES | SHOW_STATIC_OBJ_3D_PTS)) {
        unsigned char clmul = 255 / 3;
        // Apply the colormap:
        Mat colorMapImg;
        applyColorMap(depthAreaMap * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
        if(!writeIntermediateImg(colorMapImg, "static_object_depth_areas")){
            namedWindow("Static object depth areas", WINDOW_AUTOSIZE);
            imshow("Static object depth areas", colorMapImg);
            waitKey(0);
            if ((verbose & SHOW_STATIC_OBJ_DISTANCES) &&
            !(verbose & (SHOW_STATIC_OBJ_DISTANCES | SHOW_STATIC_OBJ_3D_PTS))) {
                destroyWindow("Static object depth areas");
            }
        }
    }

    //Get final depth values for each depth region
    Mat depthMapNear, depthMapMid, depthMapFar;
    getDepthMaps(depthMapNear, actUsedAreaNear, actDepthNear, actDepthMid, seedsNear, 0);
    getDepthMaps(depthMapMid, actUsedAreaMid, actDepthMid, actDepthFar, seedsMid, 1);
    getDepthMaps(depthMapFar, actUsedAreaFar, actDepthFar, maxFarDistMultiplier * actDepthFar, seedsFar, 2);

    //Combine the 3 depth maps to a single depth map
    depthMap = depthMapNear + depthMapMid + depthMapFar;

    //Visualize the depth values
    if (verbose & SHOW_STATIC_OBJ_DISTANCES) {
        Mat normalizedDepth;
        cv::normalize(depthMapNear, normalizedDepth, 0.1, 1.0, cv::NORM_MINMAX, -1, depthMapNear > 0);
        bool wtd = !writeIntermediateImg(normalizedDepth, "normalized_static_object_depth_near");
        if(wtd){
            namedWindow("Normalized Static Obj Depth Near", WINDOW_AUTOSIZE);
            imshow("Normalized Static Obj Depth Near", normalizedDepth);
        }

        normalizedDepth.release();
        cv::normalize(depthMapMid, normalizedDepth, 0.1, 1.0, cv::NORM_MINMAX, -1, depthMapMid > 0);
        wtd |= !writeIntermediateImg(normalizedDepth, "normalized_static_object_depth_mid");
        if(wtd){
            namedWindow("Normalized Static Obj Depth Mid", WINDOW_AUTOSIZE);
            imshow("Normalized Static Obj Depth Mid", normalizedDepth);
        }

        normalizedDepth.release();
        cv::normalize(depthMapFar, normalizedDepth, 0.1, 1.0, cv::NORM_MINMAX, -1, depthMapFar > 0);
        wtd |= !writeIntermediateImg(normalizedDepth, "normalized_static_object_depth_far");
        if(wtd){
            namedWindow("Normalized Static Obj Depth Far", WINDOW_AUTOSIZE);
            imshow("Normalized Static Obj Depth Far", normalizedDepth);
        }

//        Mat normalizedDepth;//, labelMask = cv::Mat::zeros(imgSize, CV_8UC1);
        //labelMask |= actUsedAreaNear | actUsedAreaMid | actUsedAreaFar;
        normalizedDepth.release();
        cv::normalize(depthMap, normalizedDepth, 0.1, 1.0, cv::NORM_MINMAX);//, -1, labelMask);
        Mat normalizedDepthColor;
        normalizedDepth.convertTo(normalizedDepthColor, CV_8UC1, 255.0);
        applyColorMap(normalizedDepthColor, normalizedDepthColor, cv::COLORMAP_RAINBOW);
        wtd |= !writeIntermediateImg(normalizedDepthColor, "normalized_static_object_depth_full");
        if(wtd){
            namedWindow("Normalized Static Obj Depth", WINDOW_AUTOSIZE);
            imshow("Normalized Static Obj Depth", normalizedDepthColor);
        }

        normalizedDepth.release();
        Mat labelMask = (depthMapFar == 0);
        cv::normalize(depthMap, normalizedDepth, 0.1, 1.0, cv::NORM_MINMAX, -1, labelMask);
        normalizedDepthColor.release();
        normalizedDepth.convertTo(normalizedDepthColor, CV_8UC1, 255.0);
        applyColorMap(normalizedDepthColor, normalizedDepthColor, cv::COLORMAP_RAINBOW);
        wtd |= !writeIntermediateImg(normalizedDepthColor, "normalized_static_object_depth_near_mid");
        if(wtd){
            namedWindow("Normalized Static Obj Depth Near and Mid", WINDOW_AUTOSIZE);
            imshow("Normalized Static Obj Depth Near and Mid", normalizedDepthColor);
        }

        //Check for 0 values
        Mat check0 = (depthMap <= 0);
        int check0val = cv::countNonZero(check0);
        if (check0val) {
            bool wtd2 = !writeIntermediateImg(check0, "zero_or_lower_obj_depth_positions");
            if(wtd2){
                namedWindow("Zero or lower Obj Depth", WINDOW_AUTOSIZE);
                imshow("Zero or lower Obj Depth", check0);
            }
            Mat checkB0 = (depthMap < 0);
            check0val = cv::countNonZero(checkB0);
            if (check0val) {
                if(!writeIntermediateImg(checkB0, "below_zero_obj_depth_positions")){
                    namedWindow("Below zero Obj Depth", WINDOW_AUTOSIZE);
                    imshow("Below zero Obj Depth", checkB0);
                    waitKey(0);
                    destroyWindow("Below zero Obj Depth");
                }
            }
            if(wtd2){
                waitKey(0);
                destroyWindow("Zero or lower Obj Depth");
            }
//            throw SequenceException("Static depth value of zero or below zero found!");
        }
        if(wtd){
            waitKey(0);
            destroyWindow("Normalized Static Obj Depth Near");
            destroyWindow("Normalized Static Obj Depth Mid");
            destroyWindow("Normalized Static Obj Depth Far");
            destroyWindow("Normalized Static Obj Depth");
            destroyWindow("Normalized Static Obj Depth Near and Mid");
        }
    }
    if (!(verbose & SHOW_STATIC_OBJ_3D_PTS)) {
        destroyAllWindows();
    }
}

//Get overlaps of filled areas (2 different) and remove them
void genStereoSequ::delOverlaps2(cv::Mat &depthArea1, cv::Mat &depthArea2) {
    Mat overlap2 = depthArea1 & depthArea2;
    int nr_overlap2 = cv::countNonZero(overlap2);
    if (nr_overlap2) {
        Mat overlap2sv = overlap2.clone();
        int overlapDel = nr_overlap2 / 2;
        removeNrFilledPixels(cv::Size(3, 3), imgSize, overlap2, overlapDel);
        Mat changeMask = ((overlap2 ^ overlap2sv) == 0);
        depthArea1 &= changeMask;
        depthArea2 &= (overlap2 == 0);
    }
}

bool genStereoSequ::fillRemainingAreas(cv::Mat &depthArea,
                                       const cv::Mat &usedAreas,
                                       int32_t areaToFill,
                                       int32_t &actualArea,
                                       cv::InputArray otherDepthA1,
                                       cv::InputArray otherDepthA2) {
    Mat mask;
    bool only1It = false;
    if (otherDepthA1.empty() || otherDepthA2.empty()) {
        mask = (usedAreas == 0);
    } else {
        Mat otherDepthA1m = otherDepthA1.getMat();
        Mat otherDepthA2m = otherDepthA2.getMat();
        mask = (usedAreas == 0) & (otherDepthA1m == 0) & (otherDepthA2m == 0);
        only1It = true;
    }

    int strElmSi = 5, cnt = 0, maxCnt = 50, strElmSiAdd = 0, strElmSiDir[2] = {0, 0};
    int32_t siAfterDil = actualArea;
    while (((!only1It && (siAfterDil < areaToFill)) || (only1It && ((siAfterDil - actualArea) == 0))) &&
           (cnt < maxCnt)) {
        cnt++;
        Mat element;
        int elSel = (int)(rand2() % 3);
        strElmSiAdd = (rand2() % INT_MAX) % strElmSi;
        strElmSiDir[0] = (int)(rand2() % 2);
        strElmSiDir[1] = (int)(rand2() % 2);
        switch (elSel) {
            case 0:
                element = cv::getStructuringElement(MORPH_ELLIPSE, Size(strElmSi + strElmSiDir[0] * strElmSiAdd,
                                                                        strElmSi + strElmSiDir[1] * strElmSiAdd));
                break;
            case 1:
                element = cv::getStructuringElement(MORPH_RECT, Size(strElmSi + strElmSiDir[0] * strElmSiAdd,
                                                                     strElmSi + strElmSiDir[1] * strElmSiAdd));
                break;
            case 2:
                element = cv::getStructuringElement(MORPH_CROSS, Size(strElmSi + strElmSiDir[0] * strElmSiAdd,
                                                                      strElmSi + strElmSiDir[1] * strElmSiAdd));
                break;
            default:
                element = cv::getStructuringElement(MORPH_ELLIPSE, Size(strElmSi, strElmSi));
                break;
        }


        strElmSi += 2;
        Mat depthAreaDilate;
        dilate(depthArea, depthAreaDilate, element);
        depthAreaDilate &= mask;
        siAfterDil = (int32_t) cv::countNonZero(depthAreaDilate);

        if (siAfterDil >= areaToFill) {
            if (siAfterDil > areaToFill) {
                int32_t diff = siAfterDil - areaToFill;
                depthAreaDilate ^= depthArea;
                removeNrFilledPixels(element.size(), imgSize, depthAreaDilate, diff);
            }
            depthArea |= depthAreaDilate;
            actualArea = areaToFill;

            return true;
        } else if (((siAfterDil - actualArea) == 0) && (cnt > 10)) {
            return false;
        } else if (siAfterDil > actualArea) {
            depthAreaDilate.copyTo(depthArea);
            actualArea = siAfterDil;

            if (only1It)
                return true;
        }
    }
    if (cnt >= maxCnt) {
        return false;
    }

    return true;
}

void genStereoSequ::removeNrFilledPixels(cv::Size delElementSi, cv::Size matSize, cv::Mat &targetMat, int32_t nrToDel) {
    cv::Size delSiEnd(matSize.width - delElementSi.width, matSize.height - delElementSi.height);
    cv::Rect delPos(0, 0, delElementSi.width, delElementSi.height);
    Mat delMask, delZeroMask = cv::Mat::zeros(delElementSi, targetMat.type());
    int32_t diff = nrToDel;
    for (int y = 0; y < delSiEnd.height; y += delElementSi.height) {
        for (int x = 0; x < delSiEnd.width; x += delElementSi.width) {
            delPos.x = x;
            delPos.y = y;
            delMask = targetMat(delPos);
            int nonZeros = cv::countNonZero(delMask);
            if (nonZeros > 0) {
                if (diff >= nonZeros) {
                    diff -= nonZeros;
                    delZeroMask.copyTo(delMask);
                } else if (diff > 0) {
                    for (int y1 = 0; y1 < delElementSi.height; y1++) {
                        for (int x1 = 0; x1 < delElementSi.width; x1++) {
                            if (delMask.at<unsigned char>(y1, x1)) {
                                delMask.at<unsigned char>(y1, x1) = 0;
                                diff--;
                                if (diff <= 0)
                                    break;
                            }
                        }
                        if (diff <= 0)
                            break;
                    }
                }
                if (diff <= 0)
                    break;
            }
        }
        if (diff <= 0)
            break;
    }
}

/*Create a random binary mask with a given size
 * mask ... Output random mask with the size  (2 * useRad) x (2 * useRad)
 * area ... Approximate area of 'ones' (255 for 8bit) in the mask
 * useRad ... Radius that should be used to fill a random circle mask
 * midR ... Circle radius in the middle of the mask that should be filled with 'ones'
 * Returns the number of 'ones' in the mask*/
int32_t genStereoSequ::getRandMask(cv::Mat &mask, int32_t area, int32_t useRad, int32_t midR) {
    int32_t usedist = 2 * useRad;
    int32_t area2 = min((int32_t) floor((double) (useRad * useRad) * M_PI), area);
    int32_t kSize = useRad / 3;
    kSize -= (kSize + 1) % 2;
    kSize = max(kSize, 3);
    if(usedist <= 3){
        mask = 255 * cv::Mat::ones(usedist, usedist, CV_8UC1);
        return usedist * usedist;
    }
    Mat mask_t = cv::Mat::zeros(usedist, usedist, CV_64FC1);
    /*Mat minVals = Mat::zeros(usedist, usedist, CV_64FC1);
    Mat maxVals = Mat::ones(usedist, usedist, CV_64FC1) * 255.0;*/
    Mat mask2, mask3;
    double mi, ma, mr;
    int actA = 0;
    do {
        do {
            randu(mask_t, cv::Scalar(0), cv::Scalar(255.0));
            cv::GaussianBlur(mask_t, mask_t, Size(kSize, kSize), 0);
            cv::minMaxLoc(mask_t, &mi, &ma);
            mr = ma - mi;
        } while (mr < 6.01);

        double mv = getRandDoubleValRng(mi + 1.0, ma - 1.0);
        double mrr = 0;
        do {
            mrr = (double) ((int)(rand2() % INT_MAX) % (int) floor((mr - 2.0) / 2.0));
        } while (nearZero(mrr));
        ma = mv + mrr;
        mi = mv - mrr;
        Mat mask_ti;
        mask_t.convertTo(mask, CV_8UC1);
        cv::threshold(mask, mask_ti, mi, 255.0, cv::THRESH_BINARY);
        cv::threshold(mask, mask, ma, 255.0, cv::THRESH_BINARY_INV);
        mask_ti.convertTo(mask2, CV_8UC1);

        /*namedWindow("rand mask thresh bin", WINDOW_AUTOSIZE);
        imshow("rand mask thresh bin", mask2);
        waitKey(0);
        namedWindow("rand mask thresh inv bin", WINDOW_AUTOSIZE);
        imshow("rand mask thresh inv bin", mask);
        waitKey(0);
        destroyWindow("rand mask thresh bin");
        destroyWindow("rand mask thresh inv bin");*/

        mask &= mask2;

        /*namedWindow("rand mask comb thresh bin", WINDOW_AUTOSIZE);
        imshow("rand mask comb thresh bin", mask);*/

        mask2 = Mat::zeros(usedist, usedist, CV_8UC1);
        cv::circle(mask2, Point(useRad - 1, useRad - 1), useRad, cv::Scalar(255), -1);
        mask &= mask2;
        mask3 = Mat::zeros(usedist, usedist, CV_8UC1);
        cv::circle(mask3, Point(useRad - 1, useRad - 1), midR, cv::Scalar(255), -1);
        mask |= mask3;

        /*namedWindow("rand mask with circles", WINDOW_AUTOSIZE);
        imshow("rand mask with circles", mask);
        waitKey(0);
        destroyWindow("rand mask comb thresh bin");
        destroyWindow("rand mask with circles");*/

        actA = cv::countNonZero(mask);
    } while (actA < 9);
    mask.copyTo(mask3);

    Mat element = cv::getStructuringElement(MORPH_CROSS, Size(3, 3));
    int maxcnt = 50;
    int minA = max(area2 / 2, 9);
    while ((actA < minA) && (maxcnt > 0)) {
        dilate(mask, mask, element);
        mask &= mask2;
        actA = cv::countNonZero(mask);
        maxcnt--;
    }
    if (maxcnt < 50) {
        /*namedWindow("rand mask dilate", WINDOW_AUTOSIZE);
        imshow("rand mask dilate", mask);
        waitKey(0);
        destroyWindow("rand mask dilate");*/

        return actA;
    }
    maxcnt = 50;
    minA = max(2 * area2 / 3, 9);
    while ((actA > minA) && (maxcnt > 0)) {
        erode(mask, mask, element);
        actA = cv::countNonZero(mask);
        maxcnt--;
    }
    if (actA == 0) {
        mask3.copyTo(mask);
        actA = cv::countNonZero(mask);
    }

    /*namedWindow("rand mask erode", WINDOW_AUTOSIZE);
    imshow("rand mask erode", mask);
    waitKey(0);
    destroyWindow("rand mask erode");*/

    return actA;
}

//Generate depth values (for every pixel) for the given areas of depth regions taking into account the depth values from backprojected 3D points
void genStereoSequ::getDepthMaps(cv::OutputArray dout, cv::Mat &din, double dmin, double dmax,
                                 std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> &initSeeds, int dNr) {
    std::vector<cv::Point3_<int32_t>> initSeedInArea;

    switch (dNr) {
        case 0:
            seedsNearFromLast = std::vector<std::vector<std::vector<cv::Point_<int32_t>>>>(3,
                                                                                           std::vector<std::vector<cv::Point_<int32_t>>>(
                                                                                                   3));
            break;
        case 1:
            seedsMidFromLast = std::vector<std::vector<std::vector<cv::Point_<int32_t>>>>(3,
                                                                                          std::vector<std::vector<cv::Point_<int32_t>>>(
                                                                                                  3));
            break;
        case 2:
            seedsFarFromLast = std::vector<std::vector<std::vector<cv::Point_<int32_t>>>>(3,
                                                                                          std::vector<std::vector<cv::Point_<int32_t>>>(
                                                                                                  3));
            break;
        default:
            break;
    }

    //Check, if there are depth seeds available that were already backprojected from 3D
    for (size_t y = 0; y < 3; y++) {
        for (size_t x = 0; x < 3; x++) {
            for (size_t i = 0; i < initSeeds[y][x].size(); i++) {
                if (initSeeds[y][x][i].z >= 0) {
                    initSeedInArea.push_back(initSeeds[y][x][i]);
                    switch (dNr) {
                        case 0:
                            seedsNearFromLast[y][x].emplace_back(
                                    cv::Point_<int32_t>(initSeedInArea.back().x, initSeedInArea.back().y));
                            break;
                        case 1:
                            seedsMidFromLast[y][x].emplace_back(
                                    cv::Point_<int32_t>(initSeedInArea.back().x, initSeedInArea.back().y));
                            break;
                        case 2:
                            seedsFarFromLast[y][x].emplace_back(
                                    cv::Point_<int32_t>(initSeedInArea.back().x, initSeedInArea.back().y));
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
    getDepthVals(dout, din, dmin, dmax, initSeedInArea);
}

//Generate depth values (for every pixel) for the given areas of depth regions
void genStereoSequ::getDepthVals(cv::OutputArray dout, const cv::Mat &din, double dmin, double dmax,
                                 std::vector<cv::Point3_<int32_t>> &initSeedInArea) {
    Mat actUsedAreaLabel;
    Mat actUsedAreaStats;
    Mat actUsedAreaCentroids;
    int nrLabels;
    vector<std::vector<double>> funcPars;
    uint16_t nL = 0;

    //Get connected areas
    nrLabels = connectedComponentsWithStats(din, actUsedAreaLabel, actUsedAreaStats, actUsedAreaCentroids, 8, CV_16U);
    nL = nrLabels;//(uint16_t) (nrLabels + 1);
    getRandDepthFuncPars(funcPars, (size_t) nL);
    //cv::ConnectedComponentsTypes::CC_STAT_HEIGHT;

    //Visualize the depth values
    if (verbose & SHOW_STATIC_OBJ_DISTANCES) {
        Mat colorMapImg;
        Mat mask = (din > 0);
        buildColorMapHSV2RGB(actUsedAreaLabel, colorMapImg, nrLabels, mask);
        if(!writeIntermediateImg(colorMapImg, "Static_Obj_Connected_Components")){
            namedWindow("Static Obj Connected Components", WINDOW_AUTOSIZE);
            imshow("Static Obj Connected Components", colorMapImg);
            waitKey(0);
            destroyWindow("Static Obj Connected Components");
        }
    }

    //dout.release();
    dout.create(imgSize, CV_64FC1);
    Mat dout_ = dout.getMat();
    dout_.setTo(Scalar(0));
//    dout = Mat::zeros(imgSize, CV_64FC1);
    vector<cv::Point> singlePixelAreas;
    for (uint16_t i = 0; i < nL; i++) {
        Rect labelBB = Rect(actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_LEFT),
                            actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_TOP),
                            actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_WIDTH),
                            actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_HEIGHT));

        if (labelBB.area() == 1) {
            singlePixelAreas.emplace_back(Point(labelBB.x, labelBB.y));
            continue;
        }

        Mat laMat = actUsedAreaLabel(labelBB);
        Mat doutSlice = dout_(labelBB);
        Mat dinSlice = din(labelBB);

        double dmin_tmp = getRandDoubleValRng(dmin, dmin + 0.6 * (dmax - dmin));
        double dmax_tmp = getRandDoubleValRng(dmin_tmp + 0.1 * (dmax - dmin), dmax);
        double drange = dmax_tmp - dmin_tmp;
        double rXr = getRandDoubleValRng(1.5, 3.0);
        double rYr = getRandDoubleValRng(1.5, 3.0);
        double h2 = (double) actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_HEIGHT);
        h2 *= h2;
        double w2 = (double) actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_WIDTH);
        w2 *= w2;
        double scale = sqrt(h2 + w2) / 2.0;
        double rXrSc = rXr / scale;
        double rYrSc = rYr / scale;
        double cx = actUsedAreaCentroids.at<double>(i, 0) -
                    (double) actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_LEFT);
        double cy = actUsedAreaCentroids.at<double>(i, 1) -
                    (double) actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_TOP);

        //If an initial seed was backprojected from 3D to this component, the depth range of the current component must be similar
        if (!initSeedInArea.empty()) {
            int32_t minX = labelBB.x;
            int32_t maxX = minX + labelBB.width;
            int32_t minY = labelBB.y;
            int32_t maxY = minY + labelBB.height;
            vector<double> initDepths;
            for (auto& j : initSeedInArea) {
                if ((j.x >= minX) && (j.x < maxX) &&
                    (j.y >= minY) && (j.y < maxY)) {
                    if (actUsedAreaLabel.at<uint16_t>(j.y, j.x) == i) {
                        initDepths.push_back(
                                actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[j.z]].z);
                    }
                }
            }
            if (!initDepths.empty()) {
                if (initDepths.size() == 1) {
                    double tmp = getRandDoubleValRng(0.05, 0.5);
                    dmin_tmp = initDepths[0] - tmp * (dmax - dmin);
                    dmax_tmp = initDepths[0] + tmp * (dmax - dmin);
                } else {
                    auto minMaxD = std::minmax_element(initDepths.begin(), initDepths.end());
                    double range1 = *minMaxD.second - *minMaxD.first;
                    if (range1 < 0.05 * (dmax - dmin)) {
                        double dmid_tmp = *minMaxD.first + range1 / 2.0;
                        double tmp = getRandDoubleValRng(0.05, 0.5);
                        dmin_tmp = dmid_tmp - tmp * (dmax - dmin);
                        dmax_tmp = dmid_tmp + tmp * (dmax - dmin);
                    } else {
                        dmin_tmp = *minMaxD.first - range1 / 2.0;
                        dmax_tmp = *minMaxD.second + range1 / 2.0;
                    }
                }
                dmin_tmp = std::max(dmin_tmp, dmin);
                dmax_tmp = std::min(dmax_tmp, dmax);
                drange = dmax_tmp - dmin_tmp;
                CV_Assert(drange > 0);
            }
        }

        double minVal = DBL_MAX, maxVal = -DBL_MAX;
        int32_t lareaCnt = 0, lareaNCnt = 2 * labelBB.width;
        for (int y = 0; y < labelBB.height; y++) {
            for (int x = 0; x < labelBB.width; x++) {
                if (laMat.at<uint16_t>(y, x) == i) {
                    if (dinSlice.at<unsigned char>(y, x) == 0) {
                        lareaNCnt--;
                        if ((lareaCnt == 0) && (lareaNCnt < 0)) {
                            lareaCnt = -1;
                            y = labelBB.height;
                            break;
                        }
                        continue;
                    }
                    lareaCnt++;
                    double val = getDepthFuncVal(funcPars[i], ((double) x - cx) * rXrSc, ((double) y - cy) * rYrSc);
                    doutSlice.at<double>(y, x) = val;
                    if (val > maxVal)
                        maxVal = val;
                    if (val < minVal)
                        minVal = val;
                }
            }
        }
        if (lareaCnt > 0) {
            double ra = maxVal - minVal;
            if(nearZero(ra)) ra = 1.0;
            scale = drange / ra;
            for (int y = 0; y < labelBB.height; y++) {
                for (int x = 0; x < labelBB.width; x++) {
                    if (laMat.at<uint16_t>(y, x) == i) {
                        double val = doutSlice.at<double>(y, x);
                        val -= minVal;
                        val *= scale;
                        val += dmin_tmp;
                        doutSlice.at<double>(y, x) = val;
                    }
                }
            }
        }
    }

    if (!singlePixelAreas.empty()) {
        Mat dout_big;
        const int extBord = 2;
        const int mSi = extBord * 2 + 1;
        Size mSi_ = Size(mSi, mSi);
        cv::copyMakeBorder(dout_, dout_big, extBord, extBord, extBord, extBord, BORDER_CONSTANT, Scalar(0));
        for (auto& i : singlePixelAreas) {
            Mat doutBigSlice = dout_big(Rect(i, mSi_));
            double dmsum = 0;
            int nrN0 = 0;
            for (int y = 0; y < mSi; ++y) {
                for (int x = 0; x < mSi; ++x) {
                    if (!nearZero(doutBigSlice.at<double>(y, x))) {
                        nrN0++;
                    }
                }
            }
            if (nrN0 == 0) {
                dmsum = actDepthMid;
            } else {
                dmsum = sum(doutBigSlice)[0];
                dmsum /= (double) nrN0;
            }
            dout_.at<double>(i) = dmsum;
        }
    }

/*    Mat normalizedDepth;
    cv::normalize(dout, normalizedDepth, 0.1, 1.0, cv::NORM_MINMAX, -1, dout > 0);
    namedWindow("Normalized Static Obj Depth One Depth", WINDOW_AUTOSIZE);
    imshow("Normalized Static Obj Depth One Depth", normalizedDepth);
    waitKey(0);
    destroyWindow("Normalized Static Obj Depth One Depth");*/
}

/*Calculates a depth value using the function
z = p1 * (p2 - x)^2 * e^(-x^2 - (y - p3)^2) - 10 * (x / p4 - x^p5 - y^p6) * e^(-x^2 - y^2) - p7 / 3 * e^(-(x + 1)^2 - y^2)
*/
inline double genStereoSequ::getDepthFuncVal(std::vector<double> &pars1, double x, double y) {
    if(nearZero(pars1[3])) pars1[3] = 1.0;
    double tmp = pars1[1] - x;
    tmp *= tmp;
    double z = pars1[0] * tmp;
    tmp = y - pars1[2];
    tmp *= -tmp;
    tmp -= x * x;
    z *= exp(tmp);
    /*double tmp1[4];
	tmp1[0] = x / pars1[3];
    tmp1[1] = std::pow(x, pars1[4]);
    tmp1[2] = std::pow(y, pars1[5]);
    tmp1[3] = exp(-x * x - y * y);
    z -= 10.0 * (tmp1[0] - tmp1[1] - tmp1[2]) * tmp1[3];*/
    z -= 10.0 * (x / pars1[3] - std::pow(x, pars1[4]) - std::pow(y, pars1[5])) * exp(-x * x - y * y);
    tmp = x + 1.0;
    tmp *= -tmp;
    z -= pars1[6] / 3.0 * exp(tmp - y * y);
    return z;
}

/*Calculate random parameters for the function generating depth values
There are 7 random paramters p:
z = p1 * (p2 - x)^2 * e^(-x^2 - (y - p3)^2) - 10 * (x / p4 - x^p5 - y^p6) * e^(-x^2 - y^2) - p7 / 3 * e^(-(x + 1)^2 - y^2)
*/
void genStereoSequ::getRandDepthFuncPars(std::vector<std::vector<double>> &pars1, size_t n_pars) {
    pars1 = std::vector<std::vector<double>>(n_pars, std::vector<double>(7, 0));

    //p1:
    std::uniform_real_distribution<double> distribution(0, 10.0);
    for (size_t i = 0; i < n_pars; i++) {
        pars1[i][0] = distribution(rand_gen);
    }
    //p2:
    distribution = std::uniform_real_distribution<double>(0, 2.0);
    for (size_t i = 0; i < n_pars; i++) {
        pars1[i][1] = distribution(rand_gen);
    }
    //p3:
    distribution = std::uniform_real_distribution<double>(0, 4.0);
    for (size_t i = 0; i < n_pars; i++) {
        pars1[i][2] = distribution(rand_gen);
    }
    //p4:
    distribution = std::uniform_real_distribution<double>(0.5, 5.0);
    for (size_t i = 0; i < n_pars; i++) {
        pars1[i][3] = distribution(rand_gen);
    }
    //p5 & p6:
    distribution = std::uniform_real_distribution<double>(2.0, 7.0);
    for (size_t i = 0; i < n_pars; i++) {
        pars1[i][4] = round(distribution(rand_gen));
        pars1[i][5] = round(distribution(rand_gen));
    }
    //p7:
    distribution = std::uniform_real_distribution<double>(1.0, 40.0);
    for (size_t i = 0; i < n_pars; i++) {
        pars1[i][6] = distribution(rand_gen);
    }
}

/* Adds a few random depth pixels near a given position (no actual depth value, but a part of a mask indicating the depth range (near, mid, far)
unsigned char pixVal	In: Value assigned to the random pixel positions
cv::Mat &imgD			In/Out: Image holding all depth ranges where the new random depth pixels should be added
cv::Mat &imgSD			In/Out: Image holding only one specific depth range where the new random depth pixels should be added
cv::Mat &mask			In: Mask for imgD and imgSD (marks backprojected moving objects (with a 1))
cv::Point_<int32_t> &startpos		In: Start position (excluding this single location) from where to start adding new depth pixels
cv::Point_<int32_t> &endpos			Out: End position where the last depth pixel was set
int32_t &addArea		In/Out: Adds the number of newly inserted pixels to the given number
int32_t &maxAreaReg		In: Maximum number of specific depth pixels per image region (9x9)
cv::Size &siM1			In: Image size -1
cv::Point_<int32_t> &initSeed	In: Initial position of the seed
cv::Rect &vROI			In: ROI were it is actually allowed to add new pixels
size_t &nrAdds			In/Out: Number of times this function was called for this depth area (including preceding calls to this function)
*/
bool genStereoSequ::addAdditionalDepth(unsigned char pixVal,
                                       cv::Mat &imgD,
                                       cv::Mat &imgSD,
                                       const cv::Mat &mask,
                                       const cv::Mat &regMask,
                                       cv::Point_<int32_t> &startpos,
                                       cv::Point_<int32_t> &endpos,
                                       int32_t &addArea,
                                       int32_t &maxAreaReg,
                                       cv::Size &siM1,
                                       cv::Point_<int32_t> initSeed,
                                       cv::Rect &vROI,
                                       size_t &nrAdds,
                                       unsigned char &usedDilate,
                                       cv::InputOutputArray neighborRegMask,
                                       unsigned char regIdx) {
    const size_t max_iter = 10000;
    const size_t midDilateCnt = 300;
    Mat neighborRegMask_;
    if (!neighborRegMask.empty()) {
        neighborRegMask_ = neighborRegMask.getMat();
    }
    //get possible directions for expanding (max. 8 possibilities) by checking the masks
    vector<int32_t> directions;
    if ((nrAdds <= max_iter) && !usedDilate && ((nrAdds % midDilateCnt != 0) || (nrAdds < midDilateCnt))) {
        if (!neighborRegMask_.empty()) {
            directions = getPossibleDirections(startpos, mask, regMask, imgD, siM1, imgSD, true, neighborRegMask, regIdx);
        }
        else{
            directions = getPossibleDirections(startpos, mask, regMask, imgD, siM1, imgSD, true);
        }
    }

    if (directions.empty() || (nrAdds > max_iter) || usedDilate || ((nrAdds % midDilateCnt == 0) && (nrAdds >=
                                                                                                     midDilateCnt)))//Dilate the label if no direction was found or there were already to many iterations
    {
        int strElmSi = 3, cnt = 0, maxCnt = 10, strElmSiAdd = 0, strElmSiDir[2] = {0, 0};
        int32_t siAfterDil = addArea;
        while (((siAfterDil - addArea) == 0) && (cnt < maxCnt)) {
            cnt++;
            Mat element;
            int elSel = (int)(rand2() % 3);
            strElmSiAdd = (int)(rand2() % INT_MAX) % strElmSi;
            strElmSiDir[0] = (int)(rand2() % 2);
            strElmSiDir[1] = (int)(rand2() % 2);
            switch (elSel) {
                case 0:
                    element = cv::getStructuringElement(MORPH_ELLIPSE, Size(strElmSi + strElmSiDir[0] * strElmSiAdd,
                                                                            strElmSi + strElmSiDir[1] * strElmSiAdd));
                    break;
                case 1:
                    element = cv::getStructuringElement(MORPH_RECT, Size(strElmSi + strElmSiDir[0] * strElmSiAdd,
                                                                         strElmSi + strElmSiDir[1] * strElmSiAdd));
                    break;
                case 2:
                    element = cv::getStructuringElement(MORPH_CROSS, Size(strElmSi + strElmSiDir[0] * strElmSiAdd,
                                                                          strElmSi + strElmSiDir[1] * strElmSiAdd));
                    break;
                default:
                    element = cv::getStructuringElement(MORPH_ELLIPSE, Size(strElmSi, strElmSi));
                    break;
            }


            strElmSi += 2;
            Mat imgSDdilate;
            Mat neighborRegMaskROI;
            Mat newImgSDROI;
            if (!neighborRegMask_.empty()) {
                newImgSDROI = imgSD(vROI) & (neighborRegMask_(vROI) == regIdx);

                dilate(newImgSDROI, imgSDdilate, element);

                /*namedWindow("specific objLabels without neighbors", WINDOW_AUTOSIZE);
                imshow("specific objLabels without neighbors", (newImgSDROI > 0));
                namedWindow("specific objLabels with neighbors", WINDOW_AUTOSIZE);
                imshow("specific objLabels with neighbors", (imgSD(vROI) > 0));*/

                imgSDdilate &= (mask(vROI) == 0) & ((imgD(vROI) == 0) | newImgSDROI);
                neighborRegMaskROI = ((imgSDdilate > 0) & Mat::ones(vROI.size(), CV_8UC1)) * regIdx;

                /*namedWindow("specific objLabels without neighbors dilated and mask", WINDOW_AUTOSIZE);
                imshow("specific objLabels without neighbors dilated and mask", (imgSDdilate > 0));*/

                siAfterDil = (int32_t) cv::countNonZero(imgSDdilate);
//                imgSDdilate |= imgSD(vROI);//Removed later on

                /*namedWindow("specific objLabels with neighbors dilated and mask", WINDOW_AUTOSIZE);
                imshow("specific objLabels with neighbors dilated and mask", (imgSDdilate > 0));

                waitKey(0);
                destroyWindow("specific objLabels without neighbors");
                destroyWindow("specific objLabels with neighbors");
                destroyWindow("specific objLabels without neighbors dilated and mask");
                destroyWindow("specific objLabels with neighbors dilated and mask");*/
            } else {
                dilate(imgSD(vROI), imgSDdilate, element);
                imgSDdilate &= (mask(vROI) == 0) & ((imgD(vROI) == 0) | (imgSD(vROI) > 0));
                siAfterDil = (int32_t) cv::countNonZero(imgSDdilate);
            }



            /*static size_t visualizeMask = 0;
            if (visualizeMask % 50 == 0) {
                Mat colorMapImg;
                // Apply the colormap:
                applyColorMap(imgD * 20, colorMapImg, cv::COLORMAP_RAINBOW);
                namedWindow("combined ObjLabels1", WINDOW_AUTOSIZE);
                imshow("combined ObjLabels1", colorMapImg);

                Mat dilImgTh;
                cv::threshold(imgSDdilate, dilImgTh, 0, 255, 0);
                namedWindow("Dilated1", WINDOW_AUTOSIZE);
                imshow("Dilated1", dilImgTh);

                Mat onlyDil = (imgSDdilate ^ imgSD(vROI)) * 20 + imgSD(vROI);
                applyColorMap(onlyDil, colorMapImg, cv::COLORMAP_HOT);
                namedWindow("Dilated", WINDOW_AUTOSIZE);
                imshow("Dilated", onlyDil);

                waitKey(0);
                destroyWindow("combined ObjLabels1");
                destroyWindow("Dilated");
                destroyWindow("Dilated1");
            }
            visualizeMask++;*/

            if (siAfterDil >= maxAreaReg) {
                if (siAfterDil > maxAreaReg) {
                    int32_t diff = siAfterDil - maxAreaReg;
                    if (!neighborRegMask_.empty()) {
                        imgSDdilate ^= newImgSDROI;
                    }else{
                        imgSDdilate ^= imgSD(vROI);
                    }
                    removeNrFilledPixels(element.size(), vROI.size(), imgSDdilate, diff);
                }
                if (!neighborRegMask_.empty()) {
                    neighborRegMaskROI = ((imgSDdilate > 0) & Mat::ones(vROI.size(), CV_8UC1)) * regIdx;
                    neighborRegMask_(vROI) |= neighborRegMaskROI;
                }
                imgSD(vROI) |= imgSDdilate;
                imgSDdilate *= pixVal;
                imgD(vROI) |= imgSDdilate;
                addArea = maxAreaReg;
                nrAdds++;
                usedDilate = 1;

                return false;
            } else if ((siAfterDil - addArea) > 0) {
                if (!neighborRegMask_.empty()) {
                    neighborRegMask_(vROI) |= neighborRegMaskROI;
                    imgSDdilate |= imgSD(vROI);
                }
                imgSDdilate.copyTo(imgSD(vROI));
                imgD(vROI) &= (imgSDdilate == 0);
                imgSDdilate *= pixVal;
                imgD(vROI) |= imgSDdilate;
                if ((directions.empty() && ((nrAdds % midDilateCnt != 0) || (nrAdds < midDilateCnt))) ||
                    (nrAdds > max_iter)) {
                    usedDilate = 1;
                }
                nrAdds++;
                addArea = siAfterDil;

                return true;
            } else if ((siAfterDil - addArea) < 0) {

                throw SequenceException(
                        "Generated depth area is smaller after dilation (and using a mask afterwards) than before!");
            }
        }
        if (cnt >= maxCnt) {
            return false;
        }
    } else {
        //Get a random direction where to add a pixel
        int diri = (int)(rand2() % directions.size());
        endpos = startpos;
        nextPosition(endpos, directions[diri]);
        //Set the pixel
        /*if (imgD.at<unsigned char>(endpos) != 0) {
            cout << "Found" << endl;
        }*/
        imgD.at<unsigned char>(endpos) = pixVal;
        /*if (imgSD.at<unsigned char>(endpos) != 0) {
            cout << "Found" << endl;
        }*/
        imgSD.at<unsigned char>(endpos) = 1;
        if (!neighborRegMask_.empty()) {
            neighborRegMask_.at<unsigned char>(endpos) = regIdx;
        }
        addArea++;
        nrAdds++;
        if (addArea >= maxAreaReg) {
            return false;
        }
        //Add additional pixels in the local neighbourhood (other possible directions) of the actual added pixel
        //and prevent adding new pixels in similar directions compared to the added one
        vector<int32_t> extension;
        if (!neighborRegMask_.empty()) {
            extension = getPossibleDirections(endpos, mask, regMask, imgD, siM1, imgSD, false, neighborRegMask, regIdx);
        }else{
            extension = getPossibleDirections(endpos, mask, regMask, imgD, siM1, imgSD, false);
        }
        if (extension.size() >
            1)//Check if we can add addition pixels without blocking the way for the next iteration
        {
            //Prevent adding additional pixels to the main direction and its immediate neighbor directions
            int32_t noExt[3];
            noExt[0] = (directions[diri] + 1) % 8;
            noExt[1] = directions[diri];
            noExt[2] = (directions[diri] + 7) % 8;
            for (auto itr = extension.rbegin(); itr != extension.rend(); itr++) {
                if ((*itr == noExt[0]) ||
                    (*itr == noExt[1]) ||
                    (*itr == noExt[2])) {
                    extension.erase(std::next(itr).base());
                }
            }
            if (extension.size() > 1) {
                //Choose a random number of additional pixels to add (based on possible directions of expansion)
                int addsi = (int)(rand2() % (extension.size() + 1));
                if (addsi) {
                    if ((addsi + addArea) > maxAreaReg) {
                        addsi = maxAreaReg - addArea;
                    }
                    const int beginExt = (int)(rand2() % extension.size());
                    for (int i = 0; i < addsi; i++) {
                        cv::Point_<int32_t> singleExt = endpos;
                        const int pos = (beginExt + i) % (int) extension.size();
                        nextPosition(singleExt, extension[pos]);
                        //Set the pixel
                        /*if (imgD.at<unsigned char>(singleExt) != 0) {
                            cout << "Found" << endl;
                        }*/
                        imgD.at<unsigned char>(singleExt) = pixVal;
                        /*if (imgSD.at<unsigned char>(singleExt) != 0) {
                            cout << "Found" << endl;
                        }*/
                        imgSD.at<unsigned char>(singleExt) = 1;
                        if (!neighborRegMask_.empty()) {
                            neighborRegMask_.at<unsigned char>(singleExt) = regIdx;
                        }
                        addArea++;
                    }
                }
                if (addArea >= maxAreaReg) {
                    return false;
                }
            }
        }
    }

    return true;
}

//Get valid directions to expand the depth area given a start position
std::vector<int32_t>
genStereoSequ::getPossibleDirections(cv::Point_<int32_t> &startpos,
                                     const cv::Mat &mask,
                                     const cv::Mat &regMask,
                                     const cv::Mat &imgD,
                                     const cv::Size &siM1,
                                     const cv::Mat &imgSD,
                                     bool escArea,
                                     cv::InputArray neighborRegMask,
                                     unsigned char regIdx) {
    static int maxFixDirChange = 8;
    int fixDirChange = 0;
    Mat directions;
    unsigned char atBorderX = 0, atBorderY = 0;
    Mat directions_dist;
    vector<int32_t> dirs;
    int32_t fixedDir = 0;
    bool dirFixed = false;
    bool inOwnArea = false;
    Mat neighborRegMaskLoc;
    if (!neighborRegMask.empty()) {
        neighborRegMaskLoc = neighborRegMask.getMat();
    }
    do {
        directions = Mat::ones(3, 3, CV_8UC1);
        directions.at<unsigned char>(1,1) = 0;
        atBorderX = 0;
        atBorderY = 0;
        if (startpos.x <= 0) {
            directions.col(0) = Mat::zeros(3, 1, CV_8UC1);
            atBorderX = 0x1;
        }
        if (startpos.x >= siM1.width) {
            directions.col(2) = Mat::zeros(3, 1, CV_8UC1);
            atBorderX = 0x2;
        }
        if (startpos.y <= 0) {
            directions.row(0) = Mat::zeros(1, 3, CV_8UC1);
            atBorderY = 0x1;
        }
        if (startpos.y >= siM1.height) {
            directions.row(2) = Mat::zeros(1, 3, CV_8UC1);
            atBorderY = 0x2;
        }

        Range irx, iry, drx, dry;
        if (atBorderX) {
            const unsigned char atBorderXn = ~atBorderX;
            const unsigned char v1 = (atBorderXn &
                    (unsigned char)0x1);//results in 0x0 (for atBorderX=0x1) or 0x1 (for atBorderX=0x2)
            const unsigned char v2 = (atBorderXn & (unsigned char)0x2) + ((atBorderX & (unsigned char)0x2)
                    >> 1);//results in 0x2 (for atBorderX=0x1) or 0x1 (for atBorderX=0x2)
            irx = Range(startpos.x - (int32_t) v1, startpos.x + (int32_t) v2);
            drx = Range((int32_t) (~v1 & 0x1), 1 + (int32_t) v2);
            if (atBorderY) {
                const unsigned char atBorderYn = ~atBorderY;
                const unsigned char v3 = (atBorderYn & 0x1);
                const unsigned char v4 = (atBorderYn & 0x2) + ((atBorderY & 0x2) >> 1);
                iry = Range(startpos.y - (int32_t) v3, startpos.y + (int32_t) v4);
                dry = Range((int32_t) (~v3 & 0x1), 1 + (int32_t) v4);
            } else {
                iry = Range(startpos.y - 1, startpos.y + 2);
                dry = Range::all();
            }
        } else if (atBorderY) {
            unsigned char atBorderYn = ~atBorderY;
            const unsigned char v3 = (atBorderYn & 0x1);
            const unsigned char v4 = (atBorderYn & 0x2) + ((atBorderY & 0x2) >> 1);
            iry = Range(startpos.y - (int32_t) v3, startpos.y + (int32_t) v4);
            irx = Range(startpos.x - 1, startpos.x + 2);
            drx = Range::all();
            dry = Range((int32_t) (~v3 & 0x1), 1 + (int32_t) v4);
        } else {
            irx = Range(startpos.x - 1, startpos.x + 2);
            iry = Range(startpos.y - 1, startpos.y + 2);
            drx = Range::all();
            dry = Range::all();
        }


        directions.copyTo(directions_dist);
        if(neighborRegMaskLoc.empty()) {
            directions(dry, drx) &= (imgD(iry, irx) == 0) & (mask(iry, irx) == 0) & (regMask(iry, irx) > 0);
        }
        else{
            directions(dry, drx) &= (imgD(iry, irx) == 0) &
                    (mask(iry, irx) == 0) &
                    (regMask(iry, irx) > 0) &
                    (neighborRegMaskLoc(iry, irx) == 0);
        }
        if ((sum(directions)[0] == 0) && escArea) {
            if(neighborRegMaskLoc.empty()) {
                directions_dist(dry, drx) &=
                        ((imgD(iry, irx) == 0) | imgSD(iry, irx)) & (mask(iry, irx) == 0) & (regMask(iry, irx) > 0);
            }
            else{
                directions_dist(dry, drx) &=
                        ((imgD(iry, irx) == 0) | imgSD(iry, irx)) &
                        (mask(iry, irx) == 0) &
                        (regMask(iry, irx) > 0) &
                                ((neighborRegMaskLoc(iry, irx) == 0) | (neighborRegMaskLoc(iry, irx) == regIdx));
            }
            if (sum(directions_dist)[0] != 0) {
                if (!dirFixed) {
                    directions_dist.copyTo(directions);
                    inOwnArea = true;
                } else {
                    cv::Point_<int32_t> localPos = cv::Point_<int32_t>(1, 1);
                    nextPosition(localPos, fixedDir);
                    if (directions_dist.at<unsigned char>(localPos) == 0) {
                        if (fixDirChange > maxFixDirChange) {
                            inOwnArea = false;
                            dirFixed = false;
                            directions = Mat::zeros(3, 3, CV_8UC1);
                        } else {
                            inOwnArea = true;
                            dirFixed = false;
                            directions_dist.copyTo(directions);
                        }
                        fixDirChange++;
                    }
                }
            } else {
                inOwnArea = false;
                dirFixed = false;
                directions = Mat::zeros(3, 3, CV_8UC1);
            }
        } else {
            dirFixed = false;
            inOwnArea = false;
        }

        if (!dirFixed) {
            dirs.clear();
            for (int32_t i = 0; i < 9; i++) {
                if (directions.at<bool>(i)) {
                    switch (i) {
                        case 0:
                            dirs.push_back(0);
                            break;
                        case 1:
                            dirs.push_back(1);
                            break;
                        case 2:
                            dirs.push_back(2);
                            break;
                        case 3:
                            dirs.push_back(7);
                            break;
                        case 5:
                            dirs.push_back(3);
                            break;
                        case 6:
                            dirs.push_back(6);
                            break;
                        case 7:
                            dirs.push_back(5);
                            break;
                        case 8:
                            dirs.push_back(4);
                            break;
                        default:
                            break;
                    }
                }
            }
        }

        if (inOwnArea && !dirs.empty()) {
            if (!dirFixed) {
                if (dirs.size() == 1) {
                    fixedDir = dirs[0];
                } else {
                    //Get a random direction where to go next
                    fixedDir = dirs[rand2() % dirs.size()];
                }
                dirFixed = true;
            }
            nextPosition(startpos, fixedDir);
        }
    } while (inOwnArea);

    return dirs;
}

void genStereoSequ::nextPosition(cv::Point_<int32_t> &position, int32_t direction) {
    switch (direction) {
        case 0://direction left up
            position.x--;
            position.y--;
            break;
        case 1://direction up
            position.y--;
            break;
        case 2://direction right up
            position.x++;
            position.y--;
            break;
        case 3://direction right
            position.x++;
            break;
        case 4://direction right down
            position.x++;
            position.y++;
            break;
        case 5://direction down
            position.y++;
            break;
        case 6://direction left down
            position.x--;
            position.y++;
            break;
        case 7://direction left
            position.x--;
            break;
        default:
            break;
    }
}

//Generates correspondences and 3D points in the camera coordinate system (including false matches) from static scene elements
void genStereoSequ::getKeypoints() {
    int32_t kSi = csurr.rows;
    int32_t posadd = (kSi - 1) / 2;

    //Mark used areas (by correspondences, TN, and moving objects) in the second image
    Mat cImg2 = Mat::zeros(imgSize.height + kSi - 1, imgSize.width + kSi - 1, CV_8UC1);
    vector<size_t> delListCorrs, delList3D;
    for (int i = 0; i < actCorrsImg2TPFromLast.cols; i++) {
        Point_<int32_t> pt((int32_t) round(actCorrsImg2TPFromLast.at<double>(0, i)),
                           (int32_t) round(actCorrsImg2TPFromLast.at<double>(1, i)));
        Mat s_tmp = cImg2(Rect(pt, Size(kSi, kSi)));
        if (s_tmp.at<unsigned char>(posadd, posadd) > 0) {
            delListCorrs.push_back((size_t)i);
            delList3D.push_back(actCorrsImg12TPFromLast_Idx[i]);
            continue;
        }
        s_tmp.at<unsigned char>(posadd, posadd) = 1;
    }

    //Delete correspondences and 3D points that were to near to each other in the 2nd image
    int TPfromLastRedu = 0;
    if (!delListCorrs.empty()) {
//        if(!checkCorr3DConsistency()){
//            throw SequenceException("Correspondences are not projections of 3D points!");
//        }
        TPfromLastRedu = (int)delListCorrs.size();
        sort(delList3D.begin(), delList3D.end(),
             [](size_t first, size_t second) { return first < second; });//Ascending order

        if (!actCorrsImg1TNFromLast_Idx.empty())//Adapt the indices for TN (single keypoints without a match)
        {
            adaptIndicesNoDel(actCorrsImg1TNFromLast_Idx, delList3D);
        }
        if (!actCorrsImg2TNFromLast_Idx.empty())//Adapt the indices for TN (single keypoints without a match)
        {
            adaptIndicesNoDel(actCorrsImg2TNFromLast_Idx, delList3D);
        }
        adaptIndicesNoDel(actCorrsImg12TPFromLast_Idx, delList3D);
        deleteVecEntriesbyIdx(actImgPointCloudFromLast, delList3D);
        deleteVecEntriesbyIdx(actCorrsImg12TPFromLast_IdxWorld, delList3D);

        sort(delListCorrs.begin(), delListCorrs.end(), [](size_t first, size_t second) { return first < second; });

        vector<size_t> delListCorrs_tmp = delListCorrs;
        TPfromLastRedu -= deletedepthCatsByIdx(seedsNearFromLast, delListCorrs_tmp, actCorrsImg1TPFromLast);
        if(TPfromLastRedu > 0){
            TPfromLastRedu -= deletedepthCatsByIdx(seedsMidFromLast, delListCorrs_tmp, actCorrsImg1TPFromLast);
        }
        if(TPfromLastRedu > 0){
            TPfromLastRedu -= deletedepthCatsByIdx(seedsFarFromLast, delListCorrs_tmp, actCorrsImg1TPFromLast);
        }
        if(TPfromLastRedu > 0){
            cout << "Keypoints from last frames which should be deleted were not found!" << endl;
        }

        deleteVecEntriesbyIdx(actCorrsImg12TPFromLast_Idx, delListCorrs);
        deleteMatEntriesByIdx(actCorrsImg1TPFromLast, delListCorrs, false);
        deleteMatEntriesByIdx(actCorrsImg2TPFromLast, delListCorrs, false);
//        if(!checkCorr3DConsistency()){
//            throw SequenceException("Correspondences are not projections of 3D points!");
//        }
    }

    int nrBPTN2 = actCorrsImg2TNFromLast.cols;
    int nrBPTN2cnt = 0;
    vector<size_t> delListTNlast;
    for (int i = 0; i < nrBPTN2; i++) {
        Point_<int32_t> pt((int32_t) round(actCorrsImg2TNFromLast.at<double>(0, i)),
                           (int32_t) round(actCorrsImg2TNFromLast.at<double>(1, i)));
        Mat s_tmp = cImg2(Rect(pt, Size(kSi, kSi)));
        if (s_tmp.at<unsigned char>(posadd, posadd) > 0) {
            delListTNlast.push_back((size_t)i);
            nrBPTN2--;
            continue;
        }
        s_tmp.at<unsigned char>(posadd, posadd) = 1;
    }
    cImg2(Rect(Point(posadd, posadd), imgSize)) |= movObjMask2All;

    if (!delListTNlast.empty()) {
        sort(delListTNlast.begin(), delListTNlast.end(), [](size_t first, size_t second) { return first < second; });
        deleteVecEntriesbyIdx(actCorrsImg2TNFromLast_Idx, delListTNlast);
        deleteMatEntriesByIdx(actCorrsImg2TNFromLast, delListTNlast, false);
    }

    //Get regions of backprojected TN in first image and mark their positions; add true negatives from backprojection to the new outlier data
    vector<vector<vector<Point2d>>> x1pTN(3, vector<vector<Point2d>>(3));
    Size rSl(imgSize.width / 3, imgSize.height / 3);
    delListTNlast.clear();
    for (int i = 0; i < actCorrsImg1TNFromLast.cols; i++) {



        Point_<int32_t> pt((int32_t) round(actCorrsImg1TNFromLast.at<double>(0, i)),
                           (int32_t) round(actCorrsImg1TNFromLast.at<double>(1, i)));
        Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
        if (s_tmp.at<unsigned char>(posadd, posadd) > 0) {
            delListTNlast.push_back((size_t)i);
            continue;
        }
        s_tmp += csurr;
//        csurr.copyTo(s_tmp);

        int yreg_idx = pt.y / rSl.height;
        yreg_idx = (yreg_idx > 2) ? 2 : yreg_idx;
        int xreg_idx = pt.x / rSl.width;
        xreg_idx = (xreg_idx > 2) ? 2 : xreg_idx;
        if(pt.y < regROIs[yreg_idx][xreg_idx].y){
            yreg_idx--;
        }
        else if(pt.y >= (regROIs[yreg_idx][xreg_idx].y + regROIs[yreg_idx][xreg_idx].height)){
            yreg_idx++;
        }
        if(pt.x < regROIs[yreg_idx][xreg_idx].x){
            xreg_idx--;
        }
        else if(pt.x >= (regROIs[yreg_idx][xreg_idx].x + regROIs[yreg_idx][xreg_idx].width)){
            xreg_idx++;
        }

        if((int)x1pTN[yreg_idx][xreg_idx].size() >= nrTrueNegRegs[actFrameCnt].at<int32_t>(yreg_idx, xreg_idx)){
            delListTNlast.push_back(i);
            s_tmp -= csurr;
            continue;
        }

        x1pTN[yreg_idx][xreg_idx].push_back(Point2d(actCorrsImg1TNFromLast.at<double>(0, i), actCorrsImg1TNFromLast.at<double>(1, i)));
    }

    if (!delListTNlast.empty()) {
        sort(delListTNlast.begin(), delListTNlast.end(), [](size_t first, size_t second) { return first < second; });
        deleteVecEntriesbyIdx(actCorrsImg1TNFromLast_Idx, delListTNlast);
        deleteMatEntriesByIdx(actCorrsImg1TNFromLast, delListTNlast, false);
    }

    //For visualization
    int dispit = 0;
    const int dispit_interval = 50;

    vector<vector<vector<Point_<int32_t>>>> corrsAllD(3, vector<vector<Point_<int32_t>>>(3));
    vector<vector<vector<Point2d>>> corrsAllD2(3, vector<vector<Point2d>>(3));
    Point_<int32_t> pt;
    Point2d pt2;
    Point3d pCam;
    vector<vector<vector<Point3d>>> p3DTPnew(3, vector<vector<Point3d>>(3));
    vector<vector<vector<Point2d>>> x1TN(3, vector<vector<Point2d>>(3));
    vector<vector<vector<Point2d>>> x2TN(3, vector<vector<Point2d>>(3));
    vector<vector<vector<double>>> x2TNdistCorr(3, vector<vector<double>>(3));
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (movObjHasArea[y][x])
                continue;

            auto nrNear = (int32_t) floor(
                    depthsPerRegion[actCorrsPRIdx][y][x].near *
                    (double) nrTruePosRegs[actFrameCnt].at<int32_t>(y, x));
            auto nrFar = (int32_t) floor(
                    depthsPerRegion[actCorrsPRIdx][y][x].far *
                    (double) nrTruePosRegs[actFrameCnt].at<int32_t>(y, x));
            int32_t nrMid = nrTruePosRegs[actFrameCnt].at<int32_t>(y, x) - nrNear - nrFar;

            CV_Assert(nrMid >= 0);

            int32_t nrTN = nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) - (int32_t) x1pTN[y][x].size();

            if(nrTN < 0){
                std::vector<size_t> delListCorrsTN;
                nrTN += deletedepthCatsByNr(x1pTN[y][x], -nrTN, actCorrsImg1TNFromLast, delListCorrsTN);
                if(!delListCorrsTN.empty()){
                    deleteVecEntriesbyIdx(actCorrsImg1TNFromLast_Idx, delListCorrsTN);
                    deleteMatEntriesByIdx(actCorrsImg1TNFromLast, delListCorrsTN, false);
                }
            }

            int32_t maxSelect = max(3 * nrTruePosRegs[actFrameCnt].at<int32_t>(y, x), 1000);
            int32_t maxSelect2 = 50;
            int32_t maxSelect3 = 50;
            int32_t maxSelect4 = 50;
            std::uniform_int_distribution<int32_t> distributionX(regROIs[y][x].x,
                                                                 regROIs[y][x].x + regROIs[y][x].width - 1);
            std::uniform_int_distribution<int32_t> distributionY(regROIs[y][x].y,
                                                                 regROIs[y][x].y + regROIs[y][x].height - 1);

            vector<Point_<int32_t>> corrsNearR, corrsMidR, corrsFarR;
            vector<Point2d> corrsNearR2, corrsMidR2, corrsFarR2;
            //vector<Point3d> p3DTPnewR, p3DTNnewR;
            vector<Point3d> p3DTPnewRNear, p3DTPnewRMid, p3DTPnewRFar;
            //vector<Point2d> x1TNR;
            corrsNearR.reserve((size_t)nrNear);
            corrsMidR.reserve((size_t)nrMid);
            corrsFarR.reserve((size_t)nrFar);
            p3DTPnew[y][x].reserve((size_t)(nrNear + nrMid + nrFar));
            corrsAllD[y][x].reserve((size_t)(nrNear + nrMid + nrFar));
            p3DTPnewRNear.reserve((size_t)nrNear);
            p3DTPnewRMid.reserve((size_t)nrNear);
            p3DTPnewRFar.reserve((size_t)nrFar);
            x1TN[y][x].reserve((size_t)nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x));
            x2TN[y][x].reserve((size_t)nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x));
            x2TNdistCorr[y][x].reserve((size_t)nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x));

            //Ckeck for backprojected correspondences
            nrNear -= (int32_t) seedsNearFromLast[y][x].size();
            nrFar -= (int32_t) seedsFarFromLast[y][x].size();
            nrMid -= (int32_t) seedsMidFromLast[y][x].size();
            if (nrNear < 0)
                nrFar += nrNear;
            if (nrFar < 0)
                nrMid += nrFar;

            //Delete backprojected correspondences if there are too much of them
            if(nrMid < 0){
                int nrToDel = -1 * (int)nrMid;
                nrToDel -= deleteBackProjTPByDepth(seedsMidFromLast[y][x], nrToDel);
                nrMid = 0;
                if(nrToDel > 0){
                    nrToDel -= deleteBackProjTPByDepth(seedsFarFromLast[y][x], nrToDel);
                    nrFar = 0;
                    if(nrToDel > 0){
                        nrToDel -= deleteBackProjTPByDepth(seedsNearFromLast[y][x], nrToDel);
                        nrNear = 0;
                    }
                }
                CV_Assert(nrToDel == 0);
            }
            if(nrNear < 0) nrNear = 0;
            if(nrFar < 0) nrFar = 0;

            int32_t nrNMF = nrNear + nrMid + nrFar;

            while (((nrNear > 0) || (nrFar > 0) || (nrMid > 0)) && (maxSelect2 > 0) && (maxSelect3 > 0) &&
                   (maxSelect4 > 0) && (nrNMF > 0)) {
                pt.x = distributionX(rand_gen);
                pt.y = distributionY(rand_gen);

                if (depthAreaMap.at<unsigned char>(pt) == 1) {
                    maxSelect--;
                    if ((nrNear <= 0) && (maxSelect >= 0)) continue;
                    //Check if coordinate is too near to existing keypoint
                    Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
                    if ((s_tmp.at<unsigned char>(posadd, posadd) > 0) ||
                        (combMovObjLabelsAll.at<unsigned char>(pt) > 0)) {
                        maxSelect++;
                        maxSelect2--;
                        continue;
                    }
                    maxSelect2 = 50;
                    //Check if it is also an inlier in the right image
                    bool isInl = checkLKPInlier(pt, pt2, pCam, depthMap);
                    if (isInl) {
                        Mat s_tmp1 = cImg2(Rect((int) round(pt2.x), (int) round(pt2.y), kSi, kSi));
                        if (s_tmp1.at<unsigned char>(posadd, posadd) > 0) {
                            maxSelect++;
                            maxSelect4--;
                            continue;
                        }
                        s_tmp1.at<unsigned char>(posadd,
                                                 posadd) = 1;//The minimum distance between keypoints in the second image is fixed to 1 for new correspondences
                        maxSelect4 = 50;
                    }
                    s_tmp += csurr;
                    if (!isInl) {
                        if (nrTN > 0) {
                            x1TN[y][x].push_back(Point2d((double) pt.x, (double) pt.y));
                            nrTN--;
                        } else {
                            maxSelect++;
                            maxSelect3--;
                            s_tmp -= csurr;
                        }
                        continue;
                    }
                    maxSelect3 = 50;
                    nrNear--;
                    nrNMF--;
                    corrsNearR.push_back(pt);
                    corrsNearR2.push_back(pt2);
                    p3DTPnewRNear.push_back(pCam);
                } else if (depthAreaMap.at<unsigned char>(pt) == 2) {
                    maxSelect--;
                    if ((nrMid <= 0) && (maxSelect >= 0)) continue;
                    //Check if coordinate is too near to existing keypoint
                    Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
                    if ((s_tmp.at<unsigned char>(posadd, posadd) > 0) ||
                        (combMovObjLabelsAll.at<unsigned char>(pt) > 0)) {
                        maxSelect++;
                        maxSelect2--;
                        continue;
                    }
                    maxSelect2 = 50;
                    //Check if it is also an inlier in the right image
                    bool isInl = checkLKPInlier(pt, pt2, pCam, depthMap);
                    if (isInl) {
                        Mat s_tmp1 = cImg2(Rect((int) round(pt2.x), (int) round(pt2.y), kSi, kSi));
                        if (s_tmp1.at<unsigned char>(posadd, posadd) > 0) {
                            maxSelect++;
                            maxSelect4--;
                            continue;
                        }
                        s_tmp1.at<unsigned char>(posadd,
                                                 posadd) = 1;//The minimum distance between keypoints in the second image is fixed to 1 for new correspondences
                        maxSelect4 = 50;
                    }
                    s_tmp += csurr;
                    if (!isInl) {
                        if (nrTN > 0) {
                            x1TN[y][x].push_back(Point2d((double) pt.x, (double) pt.y));
                            nrTN--;
                        } else {
                            maxSelect++;
                            maxSelect3--;
                            s_tmp -= csurr;
                        }
                        continue;
                    }
                    maxSelect3 = 50;
                    nrMid--;
                    nrNMF--;
                    corrsMidR.push_back(pt);
                    corrsMidR2.push_back(pt2);
                    p3DTPnewRMid.push_back(pCam);
                } else if (depthAreaMap.at<unsigned char>(pt) == 3) {
                    maxSelect--;
                    if ((nrFar <= 0) && (maxSelect >= 0)) continue;
                    //Check if coordinate is too near to existing keypoint
                    Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
                    if ((s_tmp.at<unsigned char>(posadd, posadd) > 0) ||
                        (combMovObjLabelsAll.at<unsigned char>(pt) > 0)) {
                        maxSelect++;
                        maxSelect2--;
                        continue;
                    }
                    maxSelect2 = 50;
                    //Check if it is also an inlier in the right image
                    bool isInl = checkLKPInlier(pt, pt2, pCam, depthMap);
                    if (isInl) {
                        Mat s_tmp1 = cImg2(Rect((int) round(pt2.x), (int) round(pt2.y), kSi, kSi));
                        if (s_tmp1.at<unsigned char>(posadd, posadd) > 0) {
                            maxSelect++;
                            maxSelect4--;
                            continue;
                        }
                        s_tmp1.at<unsigned char>(posadd,
                                                 posadd) = 1;//The minimum distance between keypoints in the second image is fixed to 1 for new correspondences
                        maxSelect4 = 50;
                    }
                    s_tmp += csurr;
                    if (!isInl) {
                        if (nrTN > 0) {
                            x1TN[y][x].push_back(Point2d((double) pt.x, (double) pt.y));
                            nrTN--;
                        } else {
                            maxSelect++;
                            maxSelect3--;
                            s_tmp -= csurr;
                        }
                        continue;
                    }
                    maxSelect3 = 50;
                    nrFar--;
                    nrNMF--;
                    corrsFarR.push_back(pt);
                    corrsFarR2.push_back(pt2);
                    p3DTPnewRFar.push_back(pCam);
                } else {
                    cout << "Depth area not defined! This should not happen!" << endl;
                }

                //Visualize the masks
                if (verbose & SHOW_STATIC_OBJ_CORRS_GEN) {
                    if (dispit % dispit_interval == 0) {
                        if(!writeIntermediateImg((corrsIMG > 0), "Static_Corrs_mask_img1_step_" + std::to_string(dispit)) ||
                           !writeIntermediateImg((cImg2 > 0), "Static_Corrs_mask_img2_step_" + std::to_string(dispit))){
                            namedWindow("Static Corrs mask img1", WINDOW_AUTOSIZE);
                            imshow("Static Corrs mask img1", (corrsIMG > 0));
                            namedWindow("Static Corrs mask img2", WINDOW_AUTOSIZE);
                            imshow("Static Corrs mask img2", (cImg2 > 0));
                            waitKey(0);
                            destroyWindow("Static Corrs mask img1");
                            destroyWindow("Static Corrs mask img2");
                        }
                    }
                    dispit++;
                }
            }

            size_t corrsNotVisible = x1TN[y][x].size();
            size_t foundTPCorrs = corrsNearR.size() + corrsMidR.size() + corrsFarR.size();

            //Copy 3D points and correspondences
            if (!p3DTPnewRNear.empty()) {
                //std::copy(p3DTPnewRNear.begin(), p3DTPnewRNear.end(), p3DTPnew[y][x].end());
                p3DTPnew[y][x].insert(p3DTPnew[y][x].end(), p3DTPnewRNear.begin(), p3DTPnewRNear.end());
            }
            if (!p3DTPnewRMid.empty()) {
                //std::copy(p3DTPnewRMid.begin(), p3DTPnewRMid.end(), p3DTPnew[y][x].end());
                p3DTPnew[y][x].insert(p3DTPnew[y][x].end(), p3DTPnewRMid.begin(), p3DTPnewRMid.end());
            }
            if (!p3DTPnewRFar.empty()) {
                //std::copy(p3DTPnewRFar.begin(), p3DTPnewRFar.end(), p3DTPnew[y][x].end());
                p3DTPnew[y][x].insert(p3DTPnew[y][x].end(), p3DTPnewRFar.begin(), p3DTPnewRFar.end());
            }

            if (!corrsNearR.empty()) {
                //std::copy(corrsNearR.begin(), corrsNearR.end(), corrsAllD[y][x].end());
                corrsAllD[y][x].insert(corrsAllD[y][x].end(), corrsNearR.begin(), corrsNearR.end());
            }
            if (!corrsMidR.empty()) {
                //std::copy(corrsMidR.begin(), corrsMidR.end(), corrsAllD[y][x].end());
                corrsAllD[y][x].insert(corrsAllD[y][x].end(), corrsMidR.begin(), corrsMidR.end());
            }
            if (!corrsFarR.empty()) {
                //std::copy(corrsFarR.begin(), corrsFarR.end(), corrsAllD[y][x].end());
                corrsAllD[y][x].insert(corrsAllD[y][x].end(), corrsFarR.begin(), corrsFarR.end());
            }

            if (!corrsNearR2.empty()) {
                //std::copy(corrsNearR2.begin(), corrsNearR2.end(), corrsAllD2[y][x].end());
                corrsAllD2[y][x].insert(corrsAllD2[y][x].end(), corrsNearR2.begin(), corrsNearR2.end());
            }
            if (!corrsMidR2.empty()) {
                //std::copy(corrsMidR2.begin(), corrsMidR2.end(), corrsAllD2[y][x].end());
                corrsAllD2[y][x].insert(corrsAllD2[y][x].end(), corrsMidR2.begin(), corrsMidR2.end());
            }
            if (!corrsFarR2.empty()) {
                //std::copy(corrsFarR2.begin(), corrsFarR2.end(), corrsAllD2[y][x].end());
                corrsAllD2[y][x].insert(corrsAllD2[y][x].end(), corrsFarR2.begin(), corrsFarR2.end());
            }

            //Add backprojected TN to the found ones
            if(!x1pTN[y][x].empty()){
                x1TN[y][x].insert(x1TN[y][x].end(), x1pTN[y][x].begin(), x1pTN[y][x].end());
            }

            //Generate mask for visualization before adding keypoints
            Mat dispMask;
            if ((verbose & SHOW_STATIC_OBJ_CORRS_GEN) && !x1TN[y][x].empty()) {
                dispMask = (cImg2 > 0);
            }

            //Select for true negatives in image 1 (already generated ones) true negatives in image 2
            size_t selTN2 = 0;
            if (nrBPTN2cnt < nrBPTN2)//First take backprojected TN from the second image
            {
                for (size_t i = 0; i < x1TN[y][x].size(); i++) {
                    pt2.x = actCorrsImg2TNFromLast.at<double>(0, nrBPTN2cnt);
                    pt2.y = actCorrsImg2TNFromLast.at<double>(1, nrBPTN2cnt);
                    x2TN[y][x].push_back(pt2);
                    x2TNdistCorr[y][x].push_back(fakeDistTNCorrespondences);
                    nrBPTN2cnt++;
                    selTN2++;
                    if (nrBPTN2cnt >= nrBPTN2)
                        break;
                }
            }
            std::uniform_int_distribution<int32_t> distributionX2(0, imgSize.width - 1);
            std::uniform_int_distribution<int32_t> distributionY2(0, imgSize.height - 1);
            if (selTN2 < x1TN[y][x].size())//Select the rest randomly
            {
                for (size_t i = selTN2; i < x1TN[y][x].size(); i++) {
                    int max_try = 10;
                    while (max_try > 0) {
                        pt.x = distributionX2(rand_gen);
                        pt.y = distributionY2(rand_gen);
                        Mat s_tmp = cImg2(Rect(pt, Size(kSi, kSi)));
                        if (s_tmp.at<unsigned char>(posadd, posadd) > 0) {
                            max_try--;
                            continue;
                        }
//                        csurr.copyTo(s_tmp);
                        s_tmp.at<unsigned char>(posadd, posadd) = 1;
                        x2TN[y][x].push_back(Point2d((double) pt.x, (double) pt.y));
                        x2TNdistCorr[y][x].push_back(fakeDistTNCorrespondences);
                        break;
                    }
                }
                while (x1TN[y][x].size() > x2TN[y][x].size()) {
                    Mat s_tmp = corrsIMG(Rect(Point_<int32_t>((int32_t) round(x1TN[y][x].back().x),
                                                              (int32_t) round(x1TN[y][x].back().y)), Size(kSi, kSi)));
                    s_tmp -= csurr;
                    x1TN[y][x].pop_back();
                    nrTN++;
                }
            }

            //Visualize the mask afterwards
            if ((verbose & SHOW_STATIC_OBJ_CORRS_GEN) && !x1TN[y][x].empty()) {
                if (!x2TN[y][x].empty()) {
                    Mat dispMask2 = (cImg2 > 0);
                    vector<Mat> channels;
                    Mat b = Mat::zeros(dispMask2.size(), CV_8UC1);
                    channels.push_back(b);
                    channels.push_back(dispMask);
                    channels.push_back(dispMask2);
                    Mat img3c;
                    merge(channels, img3c);
                    if(!writeIntermediateImg(img3c, "Static_rand_TN_corrs_mask_img2")){
                        namedWindow("Static rand TN Corrs mask img2", WINDOW_AUTOSIZE);
                        imshow("Static rand TN Corrs mask img2", img3c);
                        waitKey(0);
                        destroyWindow("Static rand TN Corrs mask img2");
                    }
                }
            }

            //Generate random TN in image 1
            if ((nrTN > 0) && (nrBPTN2cnt < nrBPTN2))//Take backprojected TN from the second image if available
            {
                //Generate mask for visualization before adding keypoints
                if (verbose & SHOW_STATIC_OBJ_CORRS_GEN) {
                    dispMask = (corrsIMG > 0);
                }

                int32_t nrTN_tmp = nrTN;
                for (int32_t i = 0; i < nrTN_tmp; i++) {
                    int max_try = 10;
                    while (max_try > 0) {
                        pt.x = distributionX(rand_gen);
                        pt.y = distributionY(rand_gen);
                        Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
                        if ((s_tmp.at<unsigned char>(posadd, posadd) > 0) ||
                            (combMovObjLabelsAll.at<unsigned char>(pt) > 0)) {
                            max_try--;
                            continue;
                        }
                        csurr.copyTo(s_tmp);
                        x1TN[y][x].push_back(Point2d((double) pt.x, (double) pt.y));
                        pt2.x = actCorrsImg2TNFromLast.at<double>(0, nrBPTN2cnt);
                        pt2.y = actCorrsImg2TNFromLast.at<double>(1, nrBPTN2cnt);
                        nrBPTN2cnt++;
                        x2TN[y][x].push_back(pt2);
                        x2TNdistCorr[y][x].push_back(fakeDistTNCorrespondences);
                        nrTN--;
                        break;
                    }
                    if (nrBPTN2cnt >= nrBPTN2)
                        break;
                }

                //Visualize the mask afterwards
                if (verbose & SHOW_STATIC_OBJ_CORRS_GEN) {
                    Mat dispMask2 = (cImg2 > 0);
                    vector<Mat> channels;
                    Mat b = Mat::zeros(dispMask2.size(), CV_8UC1);
                    channels.push_back(b);
                    channels.push_back(dispMask);
                    channels.push_back(dispMask2);
                    Mat img3c;
                    merge(channels, img3c);
                    if(!writeIntermediateImg(img3c, "Static_rand_TN_corrs_mask_img1")){
                        namedWindow("Static rand TN Corrs mask img1", WINDOW_AUTOSIZE);
                        imshow("Static rand TN Corrs mask img1", img3c);
                        waitKey(0);
                        destroyWindow("Static rand TN Corrs mask img1");
                    }
                }
            }

            //Get the rest of TN correspondences
            if (nrTN > 0) {
                std::vector<Point2d> x1TN_tmp, x2TN_tmp;
                std::vector<double> x2TNdistCorr_tmp;
                Mat maskImg1;
                copyMakeBorder(combMovObjLabelsAll, maskImg1, posadd, posadd, posadd, posadd, BORDER_CONSTANT,
                               Scalar(0));
                maskImg1 |= corrsIMG;

                //Generate mask for visualization before adding keypoints
                Mat dispMaskImg2;
                Mat dispMaskImg1;
                if (verbose & SHOW_STATIC_OBJ_CORRS_GEN) {
                    dispMaskImg2 = (cImg2 > 0);
                    dispMaskImg1 = (maskImg1 > 0);
                }

                nrTN = genTrueNegCorrs(nrTN, distributionX, distributionY, distributionX2, distributionY2, x1TN_tmp,
                                       x2TN_tmp, x2TNdistCorr_tmp, maskImg1, cImg2, depthMap);

                //Visualize the mask afterwards
                if (verbose & SHOW_STATIC_OBJ_CORRS_GEN) {
                    Mat dispMask2Img2 = (cImg2 > 0);
                    Mat dispMask2Img1 = (maskImg1 > 0);
                    vector<Mat> channels, channels1;
                    Mat b = Mat::zeros(dispMask2Img2.size(), CV_8UC1);
                    channels.push_back(b);
                    channels.push_back(dispMaskImg2);
                    channels.push_back(dispMask2Img2);
                    channels1.push_back(b);
                    channels1.push_back(dispMaskImg1);
                    channels1.push_back(dispMask2Img1);
                    Mat img3c, img3c1;
                    merge(channels, img3c);
                    merge(channels1, img3c1);
                    if(!writeIntermediateImg(img3c1, "Static_rand_img1_rand_img2_TN_corrs_mask_img1") ||
                       !writeIntermediateImg(img3c, "Static_rand_img1_rand_img2_TN_corrs_mask_img2")){
                        namedWindow("Static rand img1 rand img2 TN Corrs mask img1", WINDOW_AUTOSIZE);
                        imshow("Static rand img1 rand img2 TN Corrs mask img1", img3c1);
                        namedWindow("Static rand img1 rand img2 TN Corrs mask img2", WINDOW_AUTOSIZE);
                        imshow("Static rand img1 rand img2 TN Corrs mask img2", img3c);
                        waitKey(0);
                        destroyWindow("Static rand img1 rand img2 TN Corrs mask img1");
                        destroyWindow("Static rand img1 rand img2 TN Corrs mask img2");
                    }
                }

                if (!x1TN_tmp.empty()) {
                    corrsIMG(Rect(Point(posadd, posadd), imgSize)) |=
                            maskImg1(Rect(Point(posadd, posadd), imgSize)) & (combMovObjLabelsAll == 0);
                    //copy(x1TN_tmp.begin(), x1TN_tmp.end(), x1TN[y][x].end());
                    x1TN[y][x].insert(x1TN[y][x].end(), x1TN_tmp.begin(), x1TN_tmp.end());
                    //copy(x2TN_tmp.begin(), x2TN_tmp.end(), x2TN[y][x].end());
                    x2TN[y][x].insert(x2TN[y][x].end(), x2TN_tmp.begin(), x2TN_tmp.end());
                    //copy(x2TNdistCorr_tmp.begin(), x2TNdistCorr_tmp.end(), x2TNdistCorr[y][x].end());
                    x2TNdistCorr[y][x].insert(x2TNdistCorr[y][x].end(), x2TNdistCorr_tmp.begin(),
                                              x2TNdistCorr_tmp.end());
                }
            }

            //Adapt the number of TP and TN in the next region based on the remaining number of TP and TN of the current region
            adaptNRCorrespondences(nrNMF, nrTN, corrsNotVisible, foundTPCorrs, x, 0, y);
        }
    }

    //Store correspondences
    actImgPointCloud.clear();
    distTNtoReal.clear();
    size_t nrTPCorrs = 0, nrTNCorrs = 0;
    vector<vector<size_t>> nrTPperR(3, vector<size_t>(3, 0)), nrTNperR(3, vector<size_t>(3,
                                                                                         0));//For checking against given values
    for (size_t y = 0; y < 3; y++) {
        for (size_t x = 0; x < 3; x++) {
            nrTPperR[y][x] = corrsAllD[y][x].size();
            nrTPCorrs += nrTPperR[y][x];
            nrTNperR[y][x] = x1TN[y][x].size();
            nrTNCorrs += nrTNperR[y][x];
        }
    }
    actCorrsImg1TP = Mat::ones(3, (int)nrTPCorrs, CV_64FC1);
    actCorrsImg2TP = Mat::ones(3, (int)nrTPCorrs, CV_64FC1);
    actCorrsImg1TN = Mat::ones(3, (int)nrTNCorrs, CV_64FC1);
    actCorrsImg2TN = Mat::ones(3, (int)nrTNCorrs, CV_64FC1);

    int cnt = 0, cnt2 = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (!p3DTPnew[y][x].empty()) {
                //std::copy(p3DTPnew[y][x].begin(), p3DTPnew[y][x].end(), actImgPointCloud.end());
                actImgPointCloud.insert(actImgPointCloud.end(), p3DTPnew[y][x].begin(), p3DTPnew[y][x].end());
            }
            if (!x2TNdistCorr[y][x].empty()) {
                //std::copy(x2TNdistCorr[y][x].begin(), x2TNdistCorr[y][x].end(), distTNtoReal.end());
                distTNtoReal.insert(distTNtoReal.end(), x2TNdistCorr[y][x].begin(), x2TNdistCorr[y][x].end());
            }

            for (size_t i = 0; i < corrsAllD[y][x].size(); i++) {
                actCorrsImg1TP.at<double>(0, cnt) = (double) corrsAllD[y][x][i].x;
                actCorrsImg1TP.at<double>(1, cnt) = (double) corrsAllD[y][x][i].y;
                actCorrsImg2TP.at<double>(0, cnt) = corrsAllD2[y][x][i].x;
                actCorrsImg2TP.at<double>(1, cnt) = corrsAllD2[y][x][i].y;
                cnt++;
            }

            for (size_t i = 0; i < x1TN[y][x].size(); i++) {
                actCorrsImg1TN.at<double>(0, cnt2) = x1TN[y][x][i].x;
                actCorrsImg1TN.at<double>(1, cnt2) = x1TN[y][x][i].y;
                actCorrsImg2TN.at<double>(0, cnt2) = x2TN[y][x][i].x;
                actCorrsImg2TN.at<double>(1, cnt2) = x2TN[y][x][i].y;
                cnt2++;
            }
        }
    }

    //Check number of static TP and TN per region and the overall inlier ratio
    if(verbose & PRINT_WARNING_MESSAGES) {
        size_t nrCorrsR = 0, nrCorrsRGiven = 0;
        size_t nrTPCorrsAll = nrTPCorrs;
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                nrTPCorrsAll += seedsNearFromLast[y][x].size();
                nrTPCorrsAll += seedsMidFromLast[y][x].size();
                nrTPCorrsAll += seedsFarFromLast[y][x].size();
            }
        }
        nrCorrsR = nrTPCorrsAll + nrTNCorrs;
        nrCorrsRGiven = (size_t) floor(sum(nrTruePosRegs[actFrameCnt])[0] + sum(nrTrueNegRegs[actFrameCnt])[0]);
        if (nrCorrsR != nrCorrsRGiven) {
            double chRate = (double) nrCorrsR / (double) nrCorrsRGiven;
            if ((chRate < 0.90) || (chRate > 1.10)) {
                cout << "Number of correspondences on static objects is " << 100.0 * (chRate - 1.0)
                     << "% different to given values!" << endl;
                cout << "Actual #: " << nrCorrsR << " Given #: " << nrCorrsRGiven << endl;
                Mat baproTN = Mat::zeros(3, 3, CV_32SC1);
                for (int k = 0; k < 3; ++k) {
                    for (int k1 = 0; k1 < 3; ++k1) {
                        size_t allnrTPperR = nrTPperR[k][k1] +
                                             seedsNearFromLast[k][k1].size() +
                                             seedsMidFromLast[k][k1].size() +
                                             seedsFarFromLast[k][k1].size();
                        if ((int32_t) allnrTPperR != nrTruePosRegs[actFrameCnt].at<int32_t>(k, k1)) {
                            cout << "# of TP for static region (x, y): (" <<
                                 k1 <<
                                 ", " << k
                                 << ") differs by "
                                 << (int32_t) allnrTPperR - nrTruePosRegs[actFrameCnt].at<int32_t>(k, k1)
                                 <<
                                 " correspondences (Actual #: " << allnrTPperR
                                 << " Given #: " << nrTruePosRegs[actFrameCnt].at<int32_t>(k, k1) << ")"
                                 << endl;
                        }
                        if ((int32_t) nrTNperR[k][k1] != nrTrueNegRegs[actFrameCnt].at<int32_t>(k, k1)) {
                            cout << "# of TN for static region (x, y): (" <<
                                 k1 <<
                                 ", " << k
                                 << ") differs by "
                                 << (int32_t) nrTNperR[k][k1] - nrTrueNegRegs[actFrameCnt].at<int32_t>(k, k1)
                                 <<
                                 " correspondences (Actual #: " << nrTNperR[k][k1]
                                 << " Given #: " << nrTrueNegRegs[actFrameCnt].at<int32_t>(k, k1) << ")"
                                 << endl;
                        }
                    }
                }
            }
        }
        double inlRatDiffSR = (double) nrTPCorrsAll / ((double) nrCorrsR + DBL_EPSILON) - inlRat[actFrameCnt];
        double testVal = min((double) nrCorrsR / 100.0, 1.0) * inlRatDiffSR / 300.0;
        if (!nearZero(testVal)) {
            cout << "Inlier ratio of static correspondences differs from global inlier ratio (0 - 1.0) by "
                 << inlRatDiffSR << endl;
        }
    }
}

//Reduce the number of TP and TN correspondences of the next moving objects/image regions for which correspondences
// are generated based on the number of TP and TN that were not be able to generate for the current
// moving object/image region because of too less space (minimum distance between keypoints)
void genStereoSequ::adaptNRCorrespondences(int32_t nrTP,
                                           int32_t nrTN,
                                           size_t corrsNotVisible,
                                           size_t foundTPCorrs,
                                           int idx_x,
                                           int32_t nr_movObj,
                                           int y) {
    int idx_xy, maxCnt;
    vector<int32_t *> ptrTP, ptrTN;
    if (nr_movObj == 0) {
        idx_xy = 3 * y + idx_x;
        maxCnt = 9;
        for (int y_ = 0; y_ < 3; ++y_) {
            for (int x_ = 0; x_ < 3; ++x_) {
                ptrTP.push_back(&nrTruePosRegs[actFrameCnt].at<int32_t>(y_, x_));
                ptrTN.push_back(&nrTrueNegRegs[actFrameCnt].at<int32_t>(y_, x_));
            }
        }
    } else {
        idx_xy = idx_x;
        maxCnt = nr_movObj;
        for (auto it = actTPPerMovObj.begin(); it != actTPPerMovObj.end(); it++) {
            ptrTP.push_back(&(*it));
        }
        for (auto it = actTNPerMovObj.begin(); it != actTNPerMovObj.end(); it++) {
            ptrTN.push_back(&(*it));
        }
    }

    if (((nrTP <= 0) && (nrTN <= 0)) || (idx_xy >= (maxCnt - 1))) {
        return;
    }
    if (nrTP < 0) {
        nrTP = 0;
    }
    if (nrTN < 0) {
        nrTN = 0;
    }
    if((*ptrTP[idx_xy] + *ptrTN[idx_xy]) == 0)
        return;

    double reductionFactor = (double) (*ptrTP[idx_xy] - nrTP + *ptrTN[idx_xy] - nrTN) /
                             (double) (*ptrTP[idx_xy] + *ptrTN[idx_xy]);

    //incorporate fraction of not visible (in cam2) features
    if ((corrsNotVisible + foundTPCorrs) > 0) {
        reductionFactor *= (double) (corrsNotVisible + foundTPCorrs) / ((double) foundTPCorrs + 0.001);
    }
    reductionFactor = reductionFactor > 1.0 ? 1.0 : reductionFactor;
    reductionFactor = reductionFactor < 0.33 ? 1.0 : reductionFactor;
    for (int j = idx_xy + 1; j < maxCnt; ++j) {
        *ptrTP[j] = (int32_t) (round((double) (*ptrTP[j]) * reductionFactor));
        *ptrTN[j] = (int32_t) round((double) (*ptrTN[j]) * reductionFactor);
    }
    //Change the number of TP and TN to correct the overall inlier ratio of moving objects / image regions
    // (as the desired inlier ratio of the current object/region is not reached)
    int32_t next_corrs = *ptrTN[idx_xy + 1] + *ptrTP[idx_xy + 1];
    int rest = nrTP + nrTN;
    if (rest > next_corrs) {
        if ((double) next_corrs / (double) rest > 0.5) {
            *ptrTP[idx_xy + 1] = nrTP;
            *ptrTN[idx_xy + 1] = nrTN;
        } else {
            for (int j = idx_xy + 2; j < maxCnt; ++j) {
                next_corrs = *ptrTN[j] + *ptrTP[j];
                if (rest > next_corrs) {
                    if ((double) next_corrs / (double) rest > 0.5) {
                        *ptrTP[j] = nrTP;
                        *ptrTN[j] = nrTN;
                        break;
                    } else {
                        continue;
                    }
                } else {
                    reductionFactor = 1.0 - (double) rest / (double) next_corrs;
                    *ptrTP[j] = (int32_t) round((double) (*ptrTP[j]) * reductionFactor);
                    *ptrTN[j] = (int32_t) round((double) (*ptrTN[j]) * reductionFactor);
                    *ptrTP[j] += nrTP;
                    *ptrTN[j] += nrTN;
                    break;
                }
            }
        }
    } else if(next_corrs > 0){
        reductionFactor = 1.0 - (double) rest / (double) next_corrs;
        *ptrTP[idx_xy + 1] = (int32_t) round((double) (*ptrTP[idx_xy + 1]) * reductionFactor);
        *ptrTN[idx_xy + 1] = (int32_t) round((double) (*ptrTN[idx_xy + 1]) * reductionFactor);
        *ptrTP[idx_xy + 1] += nrTP;
        *ptrTN[idx_xy + 1] += nrTN;
    }
}

/*Generates a number of nrTN true negative correspondences with a given x- & y- distribution (including the range) in both
images. True negative correspondences are only created in areas where the values around the selected locations in the masks
for images 1 and 2 (img1Mask and img2Mask) are zero (indicates that there is no near neighboring correspondence).
nrTN			In: Number of true negative correspondences to create
distributionX	In: x-distribution and value range in the first image
distributionY	In: y-distribution and value range in the first image
distributionX2	In: x-distribution and value range in the second image
distributionY2	In: y-distribution and value range in the second image
x1TN			Out: True negative keypoints in the first image
x2TN			Out: True negative keypoints in the second image
x2TNdistCorr	Out: Distance of a TN keypoint in the second image to its true positive location. If the value is equal fakeDistTNCorrespondences, the TN was generated completely random.
img1Mask		In/Out: Mask marking not usable regions / areas around already selected correspondences in camera 1
img2Mask		In/Out: Mask marking not usable regions / areas around already selected correspondences in camera 2

Return value: Number of true negatives that could not be selected due to area restrictions.
*/
int32_t genStereoSequ::genTrueNegCorrs(int32_t nrTN,
                                       std::uniform_int_distribution<int32_t> &distributionX,
                                       std::uniform_int_distribution<int32_t> &distributionY,
                                       std::uniform_int_distribution<int32_t> &distributionX2,
                                       std::uniform_int_distribution<int32_t> &distributionY2,
                                       std::vector<cv::Point2d> &x1TN,
                                       std::vector<cv::Point2d> &x2TN,
                                       std::vector<double> &x2TNdistCorr,
                                       cv::Mat &img1Mask,
                                       cv::Mat &img2Mask,
                                       cv::Mat &usedDepthMap)/*,
	cv::InputArray labelMask)*/
{
    int32_t kSi = csurr.rows;
    int32_t posadd = (kSi - 1) / 2;
    std::normal_distribution<double> distributionNX2(0, max(imgSize.width / 48, 10));
    std::normal_distribution<double> distributionNY2(0, max(imgSize.width / 48, 10));
    int maxSelect2 = 75;
    int maxSelect3 = max(3 * nrTN, 500);
    Point pt;
    Point2d pt2;
    Point3d pCam;

    /*Mat optLabelMask;//Optional mask for moving objects to select only TN on moving objects in the first image
    if(labelMask.empty())
    {
        optLabelMask = Mat::ones(imgSize, CV_8UC1);
    } else
    {
        optLabelMask = labelMask.getMat();
    }*/

    while ((nrTN > 0) && (maxSelect2 > 0) && (maxSelect3 > 0)) {
        pt.x = distributionX(rand_gen);
        pt.y = distributionY(rand_gen);

        Mat s_tmp = img1Mask(Rect(pt, Size(kSi, kSi)));
        if ((s_tmp.at<unsigned char>(posadd, posadd) > 0))// || (optLabelMask.at<unsigned char>(pt) == 0))
        {
            maxSelect2--;
            continue;
        }
        maxSelect2 = 75;
        s_tmp += csurr;
        x1TN.emplace_back(Point2d((double) pt.x, (double) pt.y));
        int max_try = 10;
        double perfDist = fakeDistTNCorrespondences;
        if (!checkLKPInlier(pt, pt2, pCam,
                            usedDepthMap))//Take a random corresponding point in the second image if the reprojection is not visible to get a TN
        {
            while (max_try > 0) {
                pt.x = distributionX2(rand_gen);
                pt.y = distributionY2(rand_gen);
                Mat s_tmp1 = img2Mask(Rect(pt, Size(kSi, kSi)));
                if (s_tmp1.at<unsigned char>(posadd, posadd) > 0) {
                    max_try--;
                    continue;
                }
                //s_tmp1 += csurr;
                s_tmp1.at<unsigned char>(posadd, posadd)++;
                break;
            }
            pt2 = Point2d((double) pt.x, (double) pt.y);
        } else//Distort the reprojection in the second image to get a TN
        {
            Point2d ptd;
            while (max_try > 0) {
                int maxAtBorder = 10;
                do {
                    do {
                        do {
                            ptd.x = distributionNX2(rand_gen);
                        }while(nearZero(ptd.x));
                        ptd.x += 0.75 * ptd.x / abs(ptd.x);
                        ptd.x *= 1.5;
                        do {
                            ptd.y = distributionNY2(rand_gen);
                        }while(nearZero(ptd.y));
                        ptd.y += 0.75 * ptd.y / abs(ptd.y);
                        ptd.y *= 1.5;
                    } while ((abs(ptd.x) < 1.5) && (abs(ptd.y) < 1.5));
                    pt2 += ptd;
                    maxAtBorder--;
                } while (((pt2.x < 0) || (pt2.x > (double) (imgSize.width - 1)) ||
                          (pt2.y < 0) || (pt2.y > (double) (imgSize.height - 1))) && (maxAtBorder > 0));

                if (maxAtBorder <= 0) {
                    max_try = 0;
                    break;
                }

                Mat s_tmp1 = img2Mask(Rect((int) round(pt2.x), (int) round(pt2.y), kSi, kSi));
                if (s_tmp1.at<unsigned char>(posadd, posadd) > 0) {
                    max_try--;
                    continue;
                }
                //s_tmp1 += csurr;
                s_tmp1.at<unsigned char>(posadd, posadd)++;
                perfDist = norm(ptd);
                break;
            }
        }
        if (max_try <= 0) {
            maxSelect3--;
            x1TN.pop_back();
            s_tmp -= csurr;
            continue;
        }
        x2TN.push_back(pt2);
        x2TNdistCorr.push_back(perfDist);
        nrTN--;
    }

    return nrTN;
}

//Check, if the given point in the first camera is also visible in the second camera
//Calculates the 3D-point in the camera coordinate system and the corresponding point in the second image
bool
genStereoSequ::checkLKPInlier(cv::Point_<int32_t> pt, cv::Point2d &pt2, cv::Point3d &pCam, cv::Mat &usedDepthMap) {
    Mat x = (Mat_<double>(3, 1) << (double) pt.x, (double) pt.y, 1.0);

    double depth = usedDepthMap.at<double>(pt);

    if (depth < 0) {
        throw SequenceException("Found negative depth value!");
    }

    x = K1i * x;
    if(nearZero(x.at<double>(2))) return false;
    x *= depth / x.at<double>(2);
    pCam = Point3d(x);

    Mat x2 = K2 * (actR * x + actT);
    if(nearZero(x2.at<double>(2))) return false;
    x2 /= x2.at<double>(2);
    pt2 = Point2d(x2.rowRange(0, 2));

    if ((pt2.x < 0) || (pt2.x > (double) (imgSize.width - 1)) ||
        (pt2.y < 0) || (pt2.y > (double) (imgSize.height - 1))) {
        return false;
    }

    return true;
}

//Calculate the initial number, size, and positions of moving objects in the image
void genStereoSequ::getNrSizePosMovObj() {
    //size_t nrMovObjs;//Number of moving objects in the scene
    //cv::InputArray startPosMovObjs;//Possible starting positions of moving objects in the image (must be 3x3 boolean (CV_8UC1))
    //std::pair<double, double> relAreaRangeMovObjs;//Relative area range of moving objects. Area range relative to the image area at the beginning.

    if (pars.nrMovObjs == 0) {
        return;
    }

    if (pars.startPosMovObjs.empty() || (cv::sum(pars.startPosMovObjs)[0] == 0)) {
        startPosMovObjs = Mat::zeros(3, 3, CV_8UC1);
        while (nearZero(cv::sum(startPosMovObjs)[0])) {
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    startPosMovObjs.at<unsigned char>(y, x) = (unsigned char) (rand2() % 2);
                }
            }
        }
    } else {
        startPosMovObjs = pars.startPosMovObjs;
    }

    //Check, if the input paramters are valid and if not, adapt them
    int nrStartA = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (startPosMovObjs.at<unsigned char>(y, x)) {
                nrStartA++;
            }
        }
    }

    //Calculate the minimal usable image area
    int imgArea = imgSize.area();
    for (auto& j : stereoImgsOverlapMask) {
        int overlapArea = cv::countNonZero(j);
        if(overlapArea < imgArea){
            imgArea = overlapArea;
        }
    }
    maxOPerReg = (int) ceil((float) pars.nrMovObjs / (float) nrStartA);
    int area23 = 2 * imgArea / 3;//The moving objects should not be larger than that
    minOArea = max((int) round(pars.relAreaRangeMovObjs.first * (double) imgArea), 3);
    maxOArea = max((int) round(pars.relAreaRangeMovObjs.second * (double) imgArea), minOArea + 1);

    //The maximum image area coverd with moving objects should not exeed 2/3 of the image
    if (minOArea * (int) pars.nrMovObjs > area23) {
        adaptMinNrMovObjsAndNrMovObjs((size_t) (area23 / minOArea));
        maxOArea = minOArea;
        minOArea = minOArea / 2;
    }

    //If more than 2 seeds for moving objects are within an image region (9x9), then the all moving objects in a region should cover not more than 2/3 of the region
    //This helps to reduce the propability that during the generation of the moving objects (beginning at the seed positions) one objects blocks the generation of an other
    //For less than 3 objects per region, there shouldnt be a problem as they can grow outside an image region and the propability of blocking a different moving object is not that high
    if (maxOPerReg > 2) {
        int areaPerReg23 = area23 / 9;
        if (maxOPerReg * minOArea > areaPerReg23) {
            if (minOArea > areaPerReg23) {
                maxOArea = areaPerReg23;
                minOArea = maxOArea / 2;
                maxOPerReg = 1;
            } else {
                maxOPerReg = areaPerReg23 / minOArea;
                maxOArea = minOArea;
                minOArea = minOArea / 2;
            }
            adaptMinNrMovObjsAndNrMovObjs((size_t) (maxOPerReg * nrStartA));
        }
    } else {
        maxOPerReg = 2;
    }

    //Get the number of moving object seeds per region
    Mat useableAreas = (fracUseableTPperRegion[0] > actFracUseableTPperRegionTH) & (startPosMovObjs > 0);
    int maxMovObj = countNonZero(useableAreas) * maxOPerReg;
    int nrMovObjs_tmp = (int) pars.nrMovObjs;
    if(nrMovObjs_tmp > maxMovObj){
        nrMovObjs_tmp = maxMovObj;
    }
    Mat nrPerReg = Mat::zeros(3, 3, CV_8UC1);
    while (nrMovObjs_tmp > 0) {
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                if (startPosMovObjs.at<unsigned char>(y, x) &&
                    (maxOPerReg > (int) nrPerReg.at<unsigned char>(y, x)) &&
                        (fracUseableTPperRegion[0].at<double>(y,x) > actFracUseableTPperRegionTH)) {
                    int addit = (int)(rand2() % 2);
                    if (addit) {
                        nrPerReg.at<unsigned char>(y, x)++;
                        nrMovObjs_tmp--;
                        if (nrMovObjs_tmp == 0)
                            break;
                    }
                }
            }
            if (nrMovObjs_tmp == 0)
                break;
        }
    }

    //Get the area for each moving object
    int maxObjsArea = min(area23, maxOArea * (int) pars.nrMovObjs);
    maxOArea = maxObjsArea / (int) pars.nrMovObjs;
    std::uniform_int_distribution<int32_t> distribution((int32_t) minOArea, (int32_t) maxOArea);
    movObjAreas = vector<vector<vector<int32_t>>>(3, vector<vector<int32_t>>(3));
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            int nr_tmp = (int) nrPerReg.at<unsigned char>(y, x);
            for (int i = 0; i < nr_tmp; i++) {
                movObjAreas[y][x].push_back(distribution(rand_gen));
            }
        }
    }

    //Get seed positions
    std::vector<std::vector<std::pair<bool,cv::Rect>>> validRects;
    getValidImgRegBorders(stereoImgsOverlapMask[0], validRects);
    minODist = imgSize.height / (3 * (maxOPerReg + 1));
    movObjSeeds = vector<vector<vector<cv::Point_<int32_t>>>>(3, vector<vector<cv::Point_<int32_t>>>(3));
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            int nr_tmp = (int) nrPerReg.at<unsigned char>(y, x);
            if (nr_tmp > 0) {
                if(!validRects[y][x].first){
                    nrPerReg.at<unsigned char>(y, x) = 0;
                    movObjAreas[y][x].clear();
                    continue;
                }
                rand_gen = std::default_random_engine(
                        rand2());//Prevent getting the same starting positions for equal ranges
                std::uniform_int_distribution<int> distributionX(validRects[y][x].second.x,
                                                                 validRects[y][x].second.x + validRects[y][x].second.width - 1);
                std::uniform_int_distribution<int> distributionY(validRects[y][x].second.y,
                                                                 validRects[y][x].second.y + validRects[y][x].second.height - 1);
                cv::Point_<int32_t> seedPt(distributionX(rand_gen), distributionY(rand_gen));
                while(!checkPointValidity(stereoImgsOverlapMask[0], seedPt)){
                    seedPt = cv::Point_<int32_t>(distributionX(rand_gen), distributionY(rand_gen));
                }
                movObjSeeds[y][x].push_back(seedPt);
                nr_tmp--;
                if (nr_tmp > 0) {
                    vector<int> xposes, yposes;
                    xposes.push_back(movObjSeeds[y][x].back().x);
                    yposes.push_back(movObjSeeds[y][x].back().y);
                    while (nr_tmp > 0) {
                        vector<double> xInterVals, yInterVals;
                        vector<double> xWeights, yWeights;
                        buildDistributionRanges(xposes, yposes, x, y, xInterVals, xWeights, yInterVals, yWeights, &validRects);

                        //Create piecewise uniform distribution and get a random seed
                        piecewise_constant_distribution<double> distrPieceX(xInterVals.begin(), xInterVals.end(),
                                                                            xWeights.begin());
                        piecewise_constant_distribution<double> distrPieceY(yInterVals.begin(), yInterVals.end(),
                                                                            yWeights.begin());
                        seedPt = cv::Point_<int32_t>(cv::Point_<int32_t>((int32_t) floor(distrPieceX(rand_gen)),
                                                                         (int32_t) floor(distrPieceY(rand_gen))));
                        int max_trys = 50;
                        while(!checkPointValidity(stereoImgsOverlapMask[0], seedPt) && (max_trys > 0)){
                            seedPt = cv::Point_<int32_t>(cv::Point_<int32_t>((int32_t) floor(distrPieceX(rand_gen)),
                                                                             (int32_t) floor(distrPieceY(rand_gen))));
                            max_trys--;
                        }
                        if(max_trys <= 0){
                            for (int i = 0; i < nr_tmp; ++i) {
                                nrPerReg.at<unsigned char>(y, x)--;
                                movObjAreas[y][x].pop_back();
                            }
                            break;
                        }
                        movObjSeeds[y][x].push_back(seedPt);
                        xposes.push_back(movObjSeeds[y][x].back().x);
                        yposes.push_back(movObjSeeds[y][x].back().y);
                        nr_tmp--;
                    }
                }
            }
        }
    }
}

void genStereoSequ::getValidImgRegBorders(const cv::Mat &mask, std::vector<std::vector<std::pair<bool,cv::Rect>>> &validRects){
    validRects = std::vector<std::vector<std::pair<bool,cv::Rect>>>(3, std::vector<std::pair<bool,cv::Rect>>(3));
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            Rect validRect;
            bool usable = getValidRegBorders(mask(regROIs[y][x]), validRect);
            if(usable){
                validRect.x += regROIs[y][x].x;
                validRect.y += regROIs[y][x].y;
                validRects[y][x] = make_pair(true, validRect);
            }
            else{
                validRects[y][x] = make_pair(false, cv::Rect());
            }
        }
    }
}

void genStereoSequ::adaptMinNrMovObjsAndNrMovObjs(size_t pars_nrMovObjsNew) {
    if(pars.nrMovObjs == 0)
        return;
    float ratMinActMovObj = (float) pars.minNrMovObjs / (float) pars.nrMovObjs;
    pars.minNrMovObjs = (size_t) round(ratMinActMovObj * (float) pars_nrMovObjsNew);
    pars.minNrMovObjs = (pars.minNrMovObjs > pars_nrMovObjsNew) ? pars_nrMovObjsNew : pars.minNrMovObjs;
    pars.nrMovObjs = pars_nrMovObjsNew;
}

//Build ranges and weights for a piecewise_constant_distribution based on values calculated before
void genStereoSequ::buildDistributionRanges(std::vector<int> &xposes,
                                            std::vector<int> &yposes,
                                            int &x,
                                            int &y,
                                            std::vector<double> &xInterVals,
                                            std::vector<double> &xWeights,
                                            std::vector<double> &yInterVals,
                                            std::vector<double> &yWeights,
                                            std::vector<std::vector<std::pair<bool,cv::Rect>>> *validRects) {
    sort(xposes.begin(), xposes.end());
    sort(yposes.begin(), yposes.end());

    Rect validRect = regROIs[y][x];
    if(validRects != nullptr){
        if(!validRects->at((size_t)y)[x].first){
            throw SequenceException("Area is not valid. Cannot build a valid distribution.");
        }
        validRect = validRects->at((size_t)y)[x].second;
    }

    //Get possible selection ranges for x-values
    int start = max(xposes[0] - minODist, validRect.x);
    int maxEnd = validRect.x + validRect.width - 1;
    int xyend = min(xposes[0] + minODist, maxEnd);
    if (start == validRect.x) {
        xInterVals.push_back((double) start);
        xInterVals.push_back((double) (xposes[0] + minODist));
        xWeights.push_back(0);
    } else {
        xInterVals.push_back((double) validRect.x);
        xInterVals.push_back((double) start);
        xWeights.push_back(1.0);
        if (xyend != maxEnd) {
            xInterVals.push_back((double) xyend);
            xWeights.push_back(0);
        }
    }
    if (xyend != maxEnd) {
        for (size_t i = 1; i < xposes.size(); i++) {
            start = max(xposes[i] - minODist, (int) floor(xInterVals.back()));
            if (start != (int) floor(xInterVals.back())) {
                xInterVals.push_back((double) (xposes[i] - minODist));
                xWeights.push_back(1.0);
            }
            xyend = min(xposes[i] + minODist, maxEnd);
            if (xyend != maxEnd) {
                xInterVals.push_back((double) xyend);
                xWeights.push_back(0);
            }
        }
    }
    if (xyend != maxEnd) {
        xInterVals.push_back((double) maxEnd);
        xWeights.push_back(1.0);
    }
    //Check if we are able to select a new seed position
    double wsum = 0;
    for (auto &i: xWeights) {
        wsum += i;
    }
    if (nearZero(wsum)) {
        xWeights.clear();
        xInterVals.clear();
        if(xposes.size() > 1) {
            vector<int> xposesAndEnds;
            xposesAndEnds.push_back(validRect.x);
            xposesAndEnds.insert(xposesAndEnds.end(), xposes.begin(), xposes.end());
            xposesAndEnds.push_back(maxEnd);
            vector<int> xIntervalDiffs(xposesAndEnds.size() - 1);
            for (int i = 1; i < (int)xposesAndEnds.size(); ++i) {
                xIntervalDiffs[i-1] = xposesAndEnds[i] - xposesAndEnds[i - 1];
            }
            int maxdiff = (int)std::distance(xIntervalDiffs.begin(),
                                        std::max_element(xIntervalDiffs.begin(), xIntervalDiffs.end()));
            int thisDist = xposesAndEnds[maxdiff + 1] - xposesAndEnds[maxdiff];
            if(thisDist >= minODist) {
                xInterVals.push_back((double) (xposesAndEnds[maxdiff] + minODist / 2));
                xInterVals.push_back((double) (xposesAndEnds[maxdiff + 1] - minODist / 2));
            }
            else if(thisDist >= 3){
                thisDist /= 3;
                xInterVals.push_back((double) (xposesAndEnds[maxdiff] + thisDist));
                xInterVals.push_back((double) (xposesAndEnds[maxdiff + 1] - thisDist));
            }
            else{
                throw SequenceException("Cannot select a distribution range as the border values are too near to each other!");
            }
        }
        else{
            int xIntervalDiffs[2], idx = 0;
            xIntervalDiffs[0] = xposes[0] - validRect.x;
            xIntervalDiffs[1] = maxEnd - xposes[0];
            if(xIntervalDiffs[1] > xIntervalDiffs[0]) idx = 1;
            if(xIntervalDiffs[idx] >= 2){
                int thisDist = xIntervalDiffs[idx] / 2;
                if(idx){
                    xInterVals.push_back((double) (xposes[0] + thisDist));
                    xInterVals.push_back((double) maxEnd);
                }else{
                    xInterVals.push_back((double) validRect.x);
                    xInterVals.push_back((double) (xposes[0] - thisDist));
                }
            }
            else{
                throw SequenceException("Cannot select a distribution range as the border values are too near to each other!");
            }
        }
        xWeights.push_back(1.0);
    }

    //Get possible selection ranges for y-values
    start = max(yposes[0] - minODist, validRect.y);
    maxEnd = validRect.y + validRect.height - 1;
    xyend = min(yposes[0] + minODist, maxEnd);
    if (start == validRect.y) {
        yInterVals.push_back((double) start);
        yInterVals.push_back((double) (yposes[0] + minODist));
        yWeights.push_back(0);
    } else {
        yInterVals.push_back((double) validRect.y);
        yInterVals.push_back((double) start);
        yWeights.push_back(1.0);
        if (xyend != maxEnd) {
            yInterVals.push_back((double) xyend);
            yWeights.push_back(0);
        }
    }
    if (xyend != maxEnd) {
        for (size_t i = 1; i < yposes.size(); i++) {
            start = max(yposes[i] - minODist, (int) floor(yInterVals.back()));
            if (start != (int) floor(yInterVals.back())) {
                yInterVals.push_back((double) (yposes[i] - minODist));
                yWeights.push_back(1.0);
            }
            xyend = min(yposes[i] + minODist, maxEnd);
            if (xyend != maxEnd) {
                yInterVals.push_back((double) xyend);
                yWeights.push_back(0);
            }
        }
    }
    if (xyend != maxEnd) {
        yInterVals.push_back((double) maxEnd);
        yWeights.push_back(1.0);
    }
    //Check if we are able to select a new seed position
    wsum = 0;
    for (auto& i: yWeights) {
        wsum += i;
    }
    if (nearZero(wsum)) {
        yWeights.clear();
        yInterVals.clear();
        if(yposes.size() > 1) {
            vector<int> yposesAndEnds;
            yposesAndEnds.push_back(validRect.y);
            yposesAndEnds.insert(yposesAndEnds.end(), yposes.begin(), yposes.end());
            yposesAndEnds.push_back(maxEnd);
            vector<int> yIntervalDiffs(yposesAndEnds.size() - 1);
            for (int i = 1; i < (int)yposesAndEnds.size(); ++i) {
                yIntervalDiffs[i-1] = yposesAndEnds[i] - yposesAndEnds[i - 1];
            }
            int maxdiff = (int)std::distance(yIntervalDiffs.begin(),
                                        std::max_element(yIntervalDiffs.begin(), yIntervalDiffs.end()));
            int thisDist = yposesAndEnds[maxdiff + 1] - yposesAndEnds[maxdiff];
            if(thisDist >= minODist) {
                yInterVals.push_back((double) (yposesAndEnds[maxdiff] + minODist / 2));
                yInterVals.push_back((double) (yposesAndEnds[maxdiff + 1] - minODist / 2));
            }
            else if(thisDist >= 3){
                thisDist /= 3;
                yInterVals.push_back((double) (yposesAndEnds[maxdiff] + thisDist));
                yInterVals.push_back((double) (yposesAndEnds[maxdiff + 1] - thisDist));
            }
            else{
                throw SequenceException("Cannot select a distribution range as the border values are too near to each other!");
            }
        }
        else{
            int yIntervalDiffs[2], idx = 0;
            yIntervalDiffs[0] = yposes[0] - validRect.y;
            yIntervalDiffs[1] = maxEnd - yposes[0];
            if(yIntervalDiffs[1] > yIntervalDiffs[0]) idx = 1;
            if(yIntervalDiffs[idx] >= 2){
                int thisDist = yIntervalDiffs[idx] / 2;
                if(idx){
                    yInterVals.push_back((double) (yposes[0] + thisDist));
                    yInterVals.push_back((double) maxEnd);
                }else{
                    yInterVals.push_back((double) validRect.y);
                    yInterVals.push_back((double) (yposes[0] - thisDist));
                }
            }
            else{
                throw SequenceException("Cannot select a distribution range as the border values are too near to each other!");
            }
        }
        yWeights.push_back(1.0);
    }
}

void genStereoSequ::adaptNrBPMovObjCorrs(int32_t remSize){
    //Remove backprojected moving object correspondences based on the ratio of the number of correspondences
    CV_Assert(actCorrsOnMovObjFromLast > 0);
    size_t nr_bp_movObj = movObjMaskFromLastLargeAdd.size();
    if(remSize > (actCorrsOnMovObjFromLast - (int32_t)nr_bp_movObj * 2)){
        remSize = actCorrsOnMovObjFromLast - (int32_t)nr_bp_movObj * 2;
    }

    vector<int> nr_reduce(nr_bp_movObj);
    int sumRem = 0;
    for (size_t i = 0; i < nr_bp_movObj; ++i) {
        nr_reduce[i] = (int)round((double)remSize *
                                  (double)(movObjCorrsImg1TPFromLast[i].cols + movObjCorrsImg1TNFromLast[i].cols) /
                                  (double)actCorrsOnMovObjFromLast);
        sumRem += nr_reduce[i];
    }
    while(sumRem < remSize){
        for (size_t i = 0; i < nr_bp_movObj; ++i){
            if((movObjCorrsImg1TPFromLast[i].cols + movObjCorrsImg1TNFromLast[i].cols - nr_reduce[i]) > 2){
                nr_reduce[i]++;
                sumRem++;
                if(sumRem == remSize) break;
            }
        }
    }
    while(sumRem > remSize){
        for (size_t i = 0; i < nr_bp_movObj; ++i){
            if(nr_reduce[i] > 0){
                nr_reduce[i]--;
                sumRem--;
                if(sumRem == remSize) break;
            }
        }
    }

    actCorrsOnMovObjFromLast -= remSize;

    //Calculate number of TP and TN to remove for every moving object
    vector<pair<int,int>> nr_reduceTPTN(nr_bp_movObj);
    for (size_t i = 0; i < nr_bp_movObj; ++i){
        if(nr_reduce[i] > 0){
            CV_Assert((movObjCorrsImg1TPFromLast[i].cols + movObjCorrsImg1TNFromLast[i].cols) > 0);
            double inlrat_tmp = (double)movObjCorrsImg1TPFromLast[i].cols /
                                (double)(movObjCorrsImg1TPFromLast[i].cols + movObjCorrsImg1TNFromLast[i].cols);
            int rdTP = (int)round(inlrat_tmp * (double)nr_reduce[i]);
            while(rdTP > movObjCorrsImg1TPFromLast[i].cols)
                rdTP--;
            int rdTN = nr_reduce[i] - rdTP;
            nr_reduceTPTN[i] = make_pair(rdTP,rdTN);
        }else{
            nr_reduceTPTN[i] = make_pair(0,0);
        }
    }

    //Remove the correspondences
    cv::Size masi = Size(csurr.cols, csurr.rows);
    for (size_t i = 0; i < nr_bp_movObj; ++i){
        if(nr_reduce[i] > 0){
            //Delete TP
            int rd_tmp = 0;
            int idx = 0;
            Point pt;
            if(nr_reduceTPTN[i].first > 0) {
                rd_tmp = nr_reduceTPTN[i].first;
                idx = movObjCorrsImg1TPFromLast[i].cols - 1;
                while (rd_tmp > 0) {
                    pt = Point((int) round(movObjCorrsImg1TPFromLast[i].at<double>(0, idx)),
                               (int) round(movObjCorrsImg1TPFromLast[i].at<double>(1, idx)));
                    Mat s_tmp = movObjMaskFromLastLargeAdd[i](Rect(pt, masi));
                    s_tmp -= csurr;
                    pt = Point((int) round(movObjCorrsImg2TPFromLast[i].at<double>(0, idx)),
                               (int) round(movObjCorrsImg2TPFromLast[i].at<double>(1, idx)));
                    movObjMaskFromLast2.at<unsigned char>(pt) = 0;
                    idx--;
                    rd_tmp--;
                }

                movObjCorrsImg1TPFromLast[i] = movObjCorrsImg1TPFromLast[i].colRange(0,
                                                                                     movObjCorrsImg1TPFromLast[i].cols -
                                                                                     nr_reduceTPTN[i].first);
                movObjCorrsImg2TPFromLast[i] = movObjCorrsImg2TPFromLast[i].colRange(0,
                                                                                     movObjCorrsImg2TPFromLast[i].cols -
                                                                                     nr_reduceTPTN[i].first);
                movObjCorrsImg12TPFromLast_Idx[i].erase(movObjCorrsImg12TPFromLast_Idx[i].end() - nr_reduceTPTN[i].first,
                                                        movObjCorrsImg12TPFromLast_Idx[i].end());
            }

            //Delete TN
            if(nr_reduceTPTN[i].second > 0) {
                rd_tmp = nr_reduceTPTN[i].second;
                idx = movObjCorrsImg1TNFromLast[i].cols - 1;
                while (rd_tmp > 0) {
                    pt = Point((int) round(movObjCorrsImg1TNFromLast[i].at<double>(0, idx)),
                               (int) round(movObjCorrsImg1TNFromLast[i].at<double>(1, idx)));
                    Mat s_tmp = movObjMaskFromLastLargeAdd[i](Rect(pt, masi));
                    s_tmp -= csurr;
                    pt = Point((int) round(movObjCorrsImg2TNFromLast[i].at<double>(0, idx)),
                               (int) round(movObjCorrsImg2TNFromLast[i].at<double>(1, idx)));
                    movObjMaskFromLast2.at<unsigned char>(pt) = 0;
                    idx--;
                    rd_tmp--;
                }
                movObjDistTNtoReal[i].erase(movObjDistTNtoReal[i].end() - nr_reduceTPTN[i].second, movObjDistTNtoReal[i].end());
                movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].colRange(0,
                                                                                     movObjCorrsImg1TNFromLast[i].cols -
                                                                                     nr_reduceTPTN[i].second);
                movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].colRange(0,
                                                                                     movObjCorrsImg2TNFromLast[i].cols -
                                                                                     nr_reduceTPTN[i].second);
            }
        }
    }
}

//Generates labels of moving objects within the image and calculates the percentage of overlap for each region
//Moreover, the number of static correspondences per region is adapted and the number of correspondences on the moving objects is calculated
//mask is used to exclude areas from generating labels and must have the same size as the image; mask holds the areas from backprojected moving objects
//seeds must hold the seeding positions for generating the labels
//areas must hold the desired area for every label
//corrsOnMovObjLF must hold the number of correspondences on moving objects that were backprojected (thus the objects were created one or more frames beforehand)
//					from 3D (including TN calculated using the inlier ratio).
void
genStereoSequ::generateMovObjLabels(const cv::Mat &mask,
                                    std::vector<cv::Point_<int32_t>> &seeds,
                                    std::vector<int32_t> &areas,
                                    int32_t corrsOnMovObjLF,
                                    cv::InputArray validImgMask) {
    CV_Assert(seeds.size() == areas.size());
    CV_Assert(corrsOnMovObjLF == actCorrsOnMovObjFromLast);

    size_t nr_movObj = areas.size();

    if (nr_movObj == 0)
        actCorrsOnMovObj = corrsOnMovObjLF;
    else
        actCorrsOnMovObj = (int32_t) round(pars.CorrMovObjPort * (double) nrCorrs[actFrameCnt]);

    if (actCorrsOnMovObj <= corrsOnMovObjLF) {
        nr_movObj = 0;
        seeds.clear();
        areas.clear();
        actTruePosOnMovObj = 0;
        actTrueNegOnMovObj = 0;

        //Adapt # of correspondences on backprojected moving objects
        if(actCorrsOnMovObj < corrsOnMovObjLF){
            int32_t remSize = corrsOnMovObjLF - actCorrsOnMovObj;
            //Remove them based on the ratio of the number of correspondences
            adaptNrBPMovObjCorrs(remSize);
        }
    }

    movObjLabels.clear();
    if (nr_movObj) {
        //movObjLabels.resize(nr_movObj, cv::Mat::zeros(imgSize, CV_8UC1));
        for (size_t i = 0; i < nr_movObj; i++) {
            movObjLabels.push_back(cv::Mat::zeros(imgSize, CV_8UC1));
        }
    }
    combMovObjLabels = cv::Mat::zeros(imgSize, CV_8UC1);
    //Set seeding positions in mov. obj. label images
    for (size_t i = 0; i < nr_movObj; i++) {
        movObjLabels[i].at<unsigned char>(seeds[i]) = 1;
        combMovObjLabels.at<unsigned char>(seeds[i]) = (unsigned char) (i + 1);
    }
    Size siM1(imgSize.width - 1, imgSize.height - 1);

    Mat maskValid = mask;
    if(!validImgMask.empty()){
        maskValid = mask | (validImgMask.getMat() == 0);
    }

#define MOV_OBJ_ONLY_IN_REGIONS 0
#if MOV_OBJ_ONLY_IN_REGIONS
    vector<cv::Point_<int32_t>> objRegionIndices(nr_movObj);
for (size_t i = 0; i < nr_movObj; i++)
{
objRegionIndices[i].x = seeds[i].x / (imgSize.width / 3);
objRegionIndices[i].y = seeds[i].y / (imgSize.height / 3);
}
#else
    Rect imgArea = Rect(Point(0, 0), imgSize);//Is also useless as it covers the whole image
    Mat regMask = cv::Mat::ones(imgSize,
                                CV_8UC1);//is currently not really used (should mark the areas where moving objects can grow)
#endif
    std::vector<cv::Point_<int32_t>> startposes = seeds;
    vector<int32_t> actArea(nr_movObj, 1);
    vector<size_t> nrIterations(nr_movObj, 0);
    vector<unsigned char> dilateOps(nr_movObj, 0);
    vector<bool> objNFinished(nr_movObj, true);
    int remainObj = (int) nr_movObj;

    //Generate labels
    size_t visualizeMask = 0;
    while (remainObj > 0) {
        for (size_t i = 0; i < nr_movObj; i++) {
            if (objNFinished[i]) {
//                Mat beforeAdding = movObjLabels[i].clone();
//                int32_t Asv = actArea[i];

                if (!addAdditionalDepth((unsigned char) (i + convhullPtsObj.size() + 1),
                                        combMovObjLabels,
                                        movObjLabels[i],
                                        maskValid,
#if MOV_OBJ_ONLY_IN_REGIONS
                        regmasks[objRegionIndices[i].y][objRegionIndices[i].x],
#else
                                        regMask,
#endif
                                        startposes[i],
                                        startposes[i],
                                        actArea[i],
                                        areas[i],
                                        siM1,
                                        seeds[i],
#if MOV_OBJ_ONLY_IN_REGIONS
                        regmasksROIs[objRegionIndices[i].y][objRegionIndices[i].x],
#else
                                        imgArea,
#endif
                                        nrIterations[i],
                                        dilateOps[i])) {
                    objNFinished[i] = false;
                    remainObj--;
                }
                /*Mat afterAdding =  movObjLabels[i].clone();;
                int realAreaBeforeDil = cv::countNonZero(afterAdding);
                if(realAreaBeforeDil != actArea[i])
                {
                    cout << "Area difference: " << realAreaBeforeDil - actArea[i] << endl;
                    cout << "Area diff between last and actual values: " << actArea[i] - Asv << endl;
                    Mat addingDiff = afterAdding ^ beforeAdding;
                    namedWindow("Before", WINDOW_AUTOSIZE);
                    namedWindow("After", WINDOW_AUTOSIZE);
                    namedWindow("Diff", WINDOW_AUTOSIZE);
                    imshow("Before", (beforeAdding > 0));
                    imshow("After", (afterAdding > 0));
                    imshow("Diff", (addingDiff > 0));
                    waitKey(0);
                    destroyWindow("Before");
                    destroyWindow("After");
                    destroyWindow("Diff");
                }*/
            }
            /*Mat dilImgTh4;
            cv::threshold( movObjLabels[i], dilImgTh4, 0, 255,0 );
            namedWindow( "Dilated4", WINDOW_AUTOSIZE );
            imshow("Dilated4", dilImgTh4);
            waitKey(0);
            destroyWindow("Dilated4");*/
        }
        if (verbose & SHOW_BUILD_PROC_MOV_OBJ) {
            if (visualizeMask % 200 == 0) {
                Mat colorMapImg;
                unsigned char clmul = (unsigned char)255 / (unsigned char)nr_movObj;
                // Apply the colormap:
                applyColorMap(combMovObjLabels * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
                if(!writeIntermediateImg(colorMapImg, "moving_object_labels_build_step_" + std::to_string(visualizeMask))){
                    namedWindow("combined ObjLabels", WINDOW_AUTOSIZE);
                    imshow("combined ObjLabels", colorMapImg);

                    waitKey(0);
                    destroyWindow("combined ObjLabels");
                }
            }
            visualizeMask++;
        }
    }

    //Finally visualize the labels
    if ((nr_movObj > 0) && (verbose & (SHOW_BUILD_PROC_MOV_OBJ | SHOW_MOV_OBJ_3D_PTS))) {
        //Generate colormap for moving obejcts (every object has a different color)
        Mat colors = Mat((int)nr_movObj, 1, CV_8UC1);
        unsigned char addc = nr_movObj > 255 ? (unsigned char)255 : (unsigned char) nr_movObj;
        addc = addc < (unsigned char)2 ? (unsigned char)255 : ((unsigned char)255 / (addc - (unsigned char)1));
        colors.at<unsigned char>(0) = 0;
        for (int k = 1; k < (int)nr_movObj; ++k) {
            colors.at<unsigned char>(k) = colors.at<unsigned char>(k - 1) + addc;
        }
        Mat colormap_img;
        applyColorMap(colors, colormap_img, COLORMAP_PARULA);
        Mat labelImgRGB = Mat::zeros(imgSize, CV_8UC3);
        for (size_t i = 0; i < nr_movObj; i++) {
            for (int r = 0; r < imgSize.height; r++) {
                for (int c = 0; c < imgSize.width; c++) {
                    if (movObjLabels[i].at<unsigned char>(r, c) != 0) {
                        labelImgRGB.at<cv::Vec3b>(r, c) = colormap_img.at<cv::Vec3b>(i);
                    }
                }
            }
        }
        if(!writeIntermediateImg(labelImgRGB, "moving_object_labels_build_step_" + std::to_string(visualizeMask))){
            namedWindow("Moving object labels for point cloud comparison", WINDOW_AUTOSIZE);
            imshow("Moving object labels for point cloud comparison", labelImgRGB);

            waitKey(0);
        }
    }

    //Get bounding rectangles for the areas
    if (nr_movObj > 0) {
        movObjLabelsROIs.resize(nr_movObj);
        for (size_t i = 0; i < nr_movObj; i++) {
            Mat mask_tmp = movObjLabels[i].clone();
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            size_t dilTries = 0;
            cv::findContours(mask_tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            while ((contours.size() > 1) &&
                   (dilTries <
                    5))//Prevent the detection of multiple objects if connections between parts are too small
            {
                Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
                dilate(mask_tmp, mask_tmp, element);
                contours.clear();
                hierarchy.clear();
                cv::findContours(mask_tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                dilTries++;
            }
            if (dilTries >= 5) {
                vector<vector<Point> > contours_new(1);
                for (auto cit = contours.rbegin();
                     cit != contours.rend(); cit++) {
                    contours_new[0].insert(contours_new[0].end(), cit->begin(), cit->end());
                }
                contours = contours_new;
            }
            movObjLabelsROIs[i] = cv::boundingRect(contours[0]);
        }
    }

    //Get overlap of regions and the portion of correspondences that is covered by the moving objects
    vector<vector<double>> movObjOverlap(3, vector<double>(3, 0));
    movObjHasArea = vector<vector<bool>>(3, vector<bool>(3, false));
    vector<vector<int32_t>> movObjCorrsFromStatic(3, vector<int32_t>(3, 0));
    vector<vector<int32_t>> movObjCorrsFromStaticInv(3, vector<int32_t>(3, 0));
    int32_t absNrCorrsFromStatic = 0;
    Mat statCorrsPRegNew = Mat::zeros(3, 3, CV_32SC1);
    double oldMovObjAreaImgRat = 0;
    if ((nr_movObj == 0) && (actCorrsOnMovObj > 0)) {
        oldMovObjAreaImgRat = (double) cv::countNonZero(mask) / (double) imgSize.area();
    }
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            movObjOverlap[y][x] = (double) (cv::countNonZero(combMovObjLabels(regROIs[y][x])) +
                                            cv::countNonZero(mask(regROIs[y][x]))) /
                                  (double) (regROIs[y][x].area());
            CV_Assert(movObjOverlap[y][x] >= 0);
            if (movObjOverlap[y][x] > 0.95) {
                movObjHasArea[y][x] = true;
                movObjCorrsFromStatic[y][x] = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x);
                movObjCorrsFromStaticInv[y][x] = 0;
                absNrCorrsFromStatic += movObjCorrsFromStatic[y][x];
            } else if (nearZero(movObjOverlap[y][x])) {
                statCorrsPRegNew.at<int32_t>(y, x) = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x);
                movObjCorrsFromStatic[y][x] = 0;
                movObjCorrsFromStaticInv[y][x] = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x);
            } else {
                movObjCorrsFromStatic[y][x] = (int32_t) round(
                        (double) nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) * movObjOverlap[y][x]);
                if ((nr_movObj == 0) && (actCorrsOnMovObj > 0)) {
                    int32_t maxFromOld = corrsOnMovObjLF;
                    if(!nearZero(100.0 * oldMovObjAreaImgRat)) {
                        maxFromOld = (int32_t) round(
                                (double) corrsOnMovObjLF * movObjOverlap[y][x] / oldMovObjAreaImgRat);
                    }
                    movObjCorrsFromStatic[y][x] =
                            movObjCorrsFromStatic[y][x] > maxFromOld ? maxFromOld : movObjCorrsFromStatic[y][x];
                }
                movObjCorrsFromStaticInv[y][x] =
                        nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) - movObjCorrsFromStatic[y][x];
                absNrCorrsFromStatic += movObjCorrsFromStatic[y][x];
                statCorrsPRegNew.at<int32_t>(y, x) = movObjCorrsFromStaticInv[y][x];
            }
        }
    }

    //Check if there are too many correspondences on the moving objects
    int32_t maxCorrs = 0;
    int32_t areassum = 0;
    if (nr_movObj > 0) {
        for (auto& i : actArea) {
            areassum += i;
        }
        CV_Assert(areassum > 0);
        //reduce the initial area by reducing the radius of a circle with corresponding area by 1: are_new = area - 2*sqrt(pi)*sqrt(area)+pi
//		maxCorrs = max((int32_t)(((double)(areassum)-3.545 * sqrt((double)(areassum)+3.15)) / (1.5 * pars.minKeypDist * pars.minKeypDist)), 1);
        //reduce the initial area by reducing the radius of a circle with corresponding area by reduceRadius
        double reduceRadius = pars.minKeypDist < 3.0 ? 3.0 : pars.minKeypDist;
        double tmp = max(sqrt((double) (areassum) / M_PI) - reduceRadius, pars.minKeypDist);
        maxCorrs = max((int32_t) ((tmp * tmp * M_PI) / (enlargeKPDist * avgMaskingArea)), 1);
        if ((actCorrsOnMovObj - corrsOnMovObjLF) > maxCorrs) {
            actCorrsOnMovObj = maxCorrs + corrsOnMovObjLF;
        }

        //Check, if the areas of moving objects are valid
        if(verbose & PRINT_WARNING_MESSAGES) {
            int32_t initAsum = 0;
            for (auto& i : areas) {
                initAsum += i;
            }
            if (initAsum != areassum) {
                double areaChange = (double) areassum / (double) initAsum;
                if ((areaChange < 0.90) || (areaChange > 1.10)) {
                    cout << "Areas of moving objects are more than 5% different compared to given values." << endl;
                    for (size_t i = 0; i < areas.size(); i++) {
                        areaChange = (double) actArea[i] / (double) areas[i];
                        if (!nearZero(areaChange - 1.0)) {
                            cout << "Area " << i << " with seed position (x, y): (" << seeds[i].x << ", " << seeds[i].y
                                 <<
                                 ") differs by " << 100.0 * (areaChange - 1.0) << "% or " << actArea[i] - areas[i]
                                 << " pixels."
                                 << endl;
                        }
                    }
                }
            }
        }
    }

    if (nr_movObj > 0) {
        double areaFracStaticCorrs = (double) absNrCorrsFromStatic /
                                     (double) nrCorrs[actFrameCnt];//Fraction of correspondences which the moving objects should take because of their area
        //double r_CorrMovObjPort = round(pars.CorrMovObjPort * 100.0) / 100.0;//Fraction of correspondences the user specified for the moving objects
        double r_areaFracStaticCorrs = round(areaFracStaticCorrs * 100.0) / 100.0;
        double r_effectiveFracMovObj = round((double) actCorrsOnMovObj / (double) nrCorrs[actFrameCnt] * 100.0) /
                                       100.0;//Effective not changable fraction of correspondences on moving objects
        if (r_effectiveFracMovObj >
            r_areaFracStaticCorrs)//Remove additional static correspondences and add them to the moving objects
        {
            int32_t remStat = actCorrsOnMovObj - absNrCorrsFromStatic;
            distributeStatObjCorrsOnMovObj(remStat,
                                           absNrCorrsFromStatic,
                                           movObjCorrsFromStaticInv,
                                           statCorrsPRegNew);
        } else if (r_effectiveFracMovObj <
                   r_areaFracStaticCorrs)//Distribute a part of the correspondences from moving objects over the static elements not covered by moving objects
        {
            int32_t remMov = absNrCorrsFromStatic - actCorrsOnMovObj;
            cv::Mat mask_tmp = (combMovObjLabels == 0) | (mask == 0);
            distributeMovObjCorrsOnStatObj(remMov,
                                           absNrCorrsFromStatic,
                                           mask_tmp,
                                           movObjCorrsFromStaticInv,
                                           movObjOverlap,
                                           statCorrsPRegNew);
        }
    }

    //Set new number of static correspondences
    adaptStatNrCorrsReg(statCorrsPRegNew);

    //Check the number of overall correspondences
    if(verbose & PRINT_WARNING_MESSAGES){
        int nrCorrs1 = (int)sum(nrCorrsRegs[actFrameCnt])[0] + (int)actCorrsOnMovObj;
        if(nrCorrs1 != (int)nrCorrs[actFrameCnt]){
            cout << "Total number of correspondencs differs after partitioning them between moving and static objects by " <<
            nrCorrs1 - (int)nrCorrs[actFrameCnt] << endl;
        }
    }

    //Calculate number of correspondences on newly created moving objects
    if (nr_movObj > 0) {
        actCorrsOnMovObj -= corrsOnMovObjLF;
        if(actCorrsOnMovObj == 0){
            actTruePosOnMovObj = 0;
            actTrueNegOnMovObj = 0;
            actTPPerMovObj.clear();
            actTNPerMovObj.clear();
            movObjLabels.clear();
            combMovObjLabels = cv::Mat::zeros(imgSize, CV_8UC1);
            movObjLabelsROIs.clear();
            nr_movObj = 0;
        }
        else {
            actTruePosOnMovObj = (int32_t) round(inlRat[actFrameCnt] * (double) actCorrsOnMovObj);
            actTrueNegOnMovObj = actCorrsOnMovObj - actTruePosOnMovObj;
            actTPPerMovObj.resize(nr_movObj, 0);
            actTNPerMovObj.resize(nr_movObj, 0);
        }
        if (nr_movObj > 1) {
            //First sort the areas and begin with the smallest as rounding for the number of TP and TN for every area can lead to a larger rest of correspondences that must be taken from the last area. Thus, it should be the largest.
            vector<pair<size_t, int32_t>> actAreaIdx(nr_movObj);
            for (size_t i = 0; i < nr_movObj; i++) {
                actAreaIdx[i] = make_pair(i, actArea[i]);
            }
            sort(actAreaIdx.begin(), actAreaIdx.end(),
                 [](pair<size_t, int32_t> first, pair<size_t, int32_t> second) {
                     return first.second < second.second;
                 });
            int32_t sumTP = 0, sumTN = 0;
            for (size_t i = 0; i < nr_movObj - 1; i++) {
                auto actTP = (int32_t) round(
                        (double) actTruePosOnMovObj * (double) actAreaIdx[i].second / (double) areassum);
                auto actTN = (int32_t) round(
                        (double) actTrueNegOnMovObj * (double) actAreaIdx[i].second / (double) areassum);
                actTPPerMovObj[actAreaIdx[i].first] = actTP;
                actTNPerMovObj[actAreaIdx[i].first] = actTN;
                sumTP += actTP;
                sumTN += actTN;
            }
            int32_t restTP = actTruePosOnMovObj - sumTP;
            int32_t restTN = actTrueNegOnMovObj - sumTN;
            bool isValid = true;
            if (restTP <= 0) {
                int idx = 0;
                while ((restTP <= 0) && (idx < (int)nr_movObj - 1)) {
                    if (actTPPerMovObj[actAreaIdx[idx].first] > 1) {
                        actTPPerMovObj[actAreaIdx[idx].first]--;
                        restTP++;
                    } else {
                        idx++;
                    }
                }
                if (restTP <= 0) {
                    seeds.erase(seeds.begin() + actAreaIdx[nr_movObj - 1].first);
                    areas.erase(areas.begin() + actAreaIdx[nr_movObj - 1].first);
                    combMovObjLabels = cv::Mat::zeros(imgSize, CV_8UC1);
                    for (int j = 0; j < (int)nr_movObj - 1; ++j) {
                        unsigned char pixVal;
                        if(actAreaIdx[nr_movObj - 1].first > actAreaIdx[j].first){
                            pixVal = (unsigned char) (actAreaIdx[j].first + convhullPtsObj.size() + 1);
                        }
                        else{
                            pixVal = (unsigned char) (actAreaIdx[j].first + convhullPtsObj.size());
                        }
                        combMovObjLabels |= movObjLabels[actAreaIdx[j].first] * pixVal;
                    }
                    movObjLabels.erase(movObjLabels.begin() + actAreaIdx[nr_movObj - 1].first);
                    actTPPerMovObj.erase(actTPPerMovObj.begin() + actAreaIdx[nr_movObj - 1].first);
                    actTNPerMovObj.erase(actTNPerMovObj.begin() + actAreaIdx[nr_movObj - 1].first);
                    movObjLabelsROIs.erase(movObjLabelsROIs.begin() + actAreaIdx[nr_movObj - 1].first);
                    actArea.erase(actArea.begin() + actAreaIdx[nr_movObj - 1].first);
                    for (int j = 0; j < (int)nr_movObj - 1; ++j){
                        if(actAreaIdx[nr_movObj - 1].first < actAreaIdx[j].first){
                            actAreaIdx[j].first--;
                        }
                    }
                    nr_movObj--;
                    vector<size_t> delList;
                    for (size_t i = 0; i < nr_movObj; ++i) {
                        if(actTPPerMovObj[i] <= 0){
                            delList.push_back(i);
                        }else if((actTPPerMovObj[i] == 1) && ((restTP < 0))){
                            restTP++;
                            delList.push_back(i);
                        }
                    }
                    if(!delList.empty()){
                        combMovObjLabels = cv::Mat::zeros(imgSize, CV_8UC1);
                        unsigned char delCnt = 0;
                        for (unsigned char j = 0; j < (unsigned char)nr_movObj; ++j) {
                            if(j == (unsigned char)delList[delCnt]){
                                delCnt++;
                                continue;
                            }
                            unsigned char pixVal = j - delCnt + (unsigned char) (convhullPtsObj.size() + 1);
                            combMovObjLabels |= movObjLabels[actAreaIdx[j].first] * pixVal;
                        }
                        for(int i = (int)delList.size() - 1; i >= 0; i--){
                            seeds.erase(seeds.begin() + delList[i]);
                            areas.erase(areas.begin() + delList[i]);
                            movObjLabels.erase(movObjLabels.begin() + delList[i]);
                            actTPPerMovObj.erase(actTPPerMovObj.begin() + delList[i]);
                            actTNPerMovObj.erase(actTNPerMovObj.begin() + delList[i]);
                            movObjLabelsROIs.erase(movObjLabelsROIs.begin() + delList[i]);
                            actArea.erase(actArea.begin() + delList[i]);
                            nr_movObj--;
                        }
                    }
                    if(nr_movObj == 0){
                        actCorrsOnMovObj = 0;
                        actTruePosOnMovObj = 0;
                        actTrueNegOnMovObj = 0;
                        actTPPerMovObj.clear();
                        actTNPerMovObj.clear();
                        movObjLabels.clear();
                        combMovObjLabels = cv::Mat::zeros(imgSize, CV_8UC1);
                        movObjLabelsROIs.clear();
                        isValid = false;
                    }
                    else if(nr_movObj == 1){
                        actTPPerMovObj[0] = actTruePosOnMovObj;
                        actTNPerMovObj[0] = actTrueNegOnMovObj;
                        isValid = false;
                    }
                    else{
                        areassum = 0;
                        for (auto& i : actArea) {
                            areassum += i;
                        }
                        actAreaIdx = vector<pair<size_t, int32_t>>(nr_movObj);
                        for (size_t i = 0; i < nr_movObj; i++) {
                            actAreaIdx[i] = make_pair(i, actArea[i]);
                        }
                        sort(actAreaIdx.begin(), actAreaIdx.end(),
                             [](pair<size_t, int32_t> first, pair<size_t, int32_t> second) {
                                 return first.second < second.second;
                             });
                        sumTN = 0;
                        for (size_t i = 0; i < nr_movObj - 1; i++) {
                            auto actTN = (int32_t) round(
                                    (double) actTrueNegOnMovObj * (double) actAreaIdx[i].second / (double) areassum);
                            actTNPerMovObj[actAreaIdx[i].first] = actTN;
                            sumTN += actTN;
                        }
                        restTN = actTrueNegOnMovObj - sumTN;
                    }

                } else {
                    actTPPerMovObj[actAreaIdx[nr_movObj - 1].first] = restTP;
                }
            } else {
                actTPPerMovObj[actAreaIdx[nr_movObj - 1].first] = restTP;
            }
            if (isValid) {
                if (restTN < 0) {
                    actTNPerMovObj[actAreaIdx[nr_movObj - 1].first] = 0;
                    int idx = 0;
                    while ((restTN < 0) && (idx < (int)nr_movObj - 1)) {
                        if (actTNPerMovObj[actAreaIdx[idx].first] > 0) {
                            actTNPerMovObj[actAreaIdx[idx].first]--;
                            restTN++;
                        } else {
                            idx++;
                        }
                    }
                    if (restTN < 0) {
                        throw SequenceException("Found a negative number of TN for a moving object!");
                    }
                } else {
                    actTNPerMovObj[actAreaIdx[nr_movObj - 1].first] = restTN;
                }
            }
        } else if (nr_movObj > 0){
            actTPPerMovObj[0] = actTruePosOnMovObj;
            actTNPerMovObj[0] = actTrueNegOnMovObj;
        }
    } else {
        actCorrsOnMovObj = 0;
        actTruePosOnMovObj = 0;
        actTrueNegOnMovObj = 0;
        actTPPerMovObj.clear();
        actTNPerMovObj.clear();
        movObjLabels.clear();
        combMovObjLabels = cv::Mat::zeros(imgSize, CV_8UC1);
        movObjLabelsROIs.clear();
    }

    //Combine existing and new labels of moving objects
    combMovObjLabelsAll = combMovObjLabels | movObjMaskFromLast;

    //Check the inlier ratio, TP, TN, and number of correspondences
    if((nr_movObj > 0) && (verbose & PRINT_WARNING_MESSAGES)) {
        int32_t sumTPMO = 0, sumTNMO = 0, sumCorrsMO = 0;
        for (auto& i:actTPPerMovObj) {
            sumTPMO += i;
        }
        for (auto& i:actTNPerMovObj) {
            sumTNMO += i;
        }
        sumCorrsMO = sumTPMO + sumTNMO;
        if (sumCorrsMO != actCorrsOnMovObj) {
            cout << "Sum of number of correspondences on moving objects is different to given number." << endl;
            if (sumTPMO != actTruePosOnMovObj) {
                cout << "Sum of number of TP correspondences on moving objects is different to given number. Sum: " <<
                     sumTPMO << " Given: " << actTruePosOnMovObj << endl;
            }
            if (sumTNMO != actTrueNegOnMovObj) {
                cout << "Sum of number of TN correspondences on moving objects is different to given number. Sum: " <<
                     sumTNMO << " Given: " << actTrueNegOnMovObj << endl;
            }
        }
        double inlRatDiffMO = (double) sumTPMO / (double) sumCorrsMO - inlRat[actFrameCnt];
        double testVal = min((double) sumCorrsMO / 100.0, 1.0) * inlRatDiffMO / 300.0;
        if (!nearZero(testVal)) {
            cout << "Inlier ratio of moving object correspondences differs from global inlier ratio (0 - 1.0) by "
                 << inlRatDiffMO << endl;
        }

        //Check the overall inlier ratio
        double tps = (double) sum(nrTruePosRegs[actFrameCnt])[0] + sumTPMO;
        double nrCorrs1 = (double) sum(nrCorrsRegs[actFrameCnt])[0] + sumCorrsMO;
        double inlRatDiffSR = tps / (nrCorrs1 + DBL_EPSILON) - inlRat[actFrameCnt];
        testVal = min(nrCorrs1 / 100.0, 1.0) * inlRatDiffSR / 300.0;
        if (!nearZero(testVal)) {
            cout << "Inlier ratio of combined static and moving correspondences after changing it because of moving objects differs "
                    "from global inlier ratio (0 - 1.0) by "
                 << inlRatDiffSR << ". THIS SHOULD NOT HAPPEN!" << endl;
        }
    }

}

void genStereoSequ::adaptStatNrCorrsReg(const cv::Mat &statCorrsPRegNew){
    CV_Assert((statCorrsPRegNew.cols == 3) && (statCorrsPRegNew.rows == 3) && (statCorrsPRegNew.type() == CV_32SC1));

    //Calculate the TN per region first as this is an indicator for regions where the camera views of the 2 stereo
    // cameras are not intersecting
    Mat TNdistr, TNdistr1;
    nrTrueNegRegs[actFrameCnt].convertTo(TNdistr, CV_64FC1);
    Mat areaFull = Mat::zeros(3, 3, CV_8UC1);
    double sumTN = sum(TNdistr)[0];
    if(!nearZero(sumTN)) {
        TNdistr /= sumTN;
        double allRegCorrsNew = (double) sum(statCorrsPRegNew)[0];
        double nrTPallReg = allRegCorrsNew * inlRat[actFrameCnt];
        double nrTNallReg = allRegCorrsNew - nrTPallReg;
        TNdistr1 = TNdistr * nrTNallReg;
        nrTrueNegRegs[actFrameCnt].release();
        TNdistr1.convertTo(nrTrueNegRegs[actFrameCnt], CV_32SC1, 1.0, 0.5);//Corresponds to round
        nrTrueNegRegs[actFrameCnt].copyTo(TNdistr1);
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                if (nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) > statCorrsPRegNew.at<int32_t>(y, x)) {
                    nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) = statCorrsPRegNew.at<int32_t>(y, x);
                    areaFull.at<bool>(y, x) = true;
                }
            }
        }
    }

    auto remStatrem = (int32_t)round(sum(TNdistr1)[0] - sum(nrTrueNegRegs[actFrameCnt])[0]);
    if(remStatrem > 0) {
        int32_t remStat = remStatrem;
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                if (!areaFull.at<bool>(y, x) && (remStatrem > 0)) {
                    auto val = (int32_t) round(
                            TNdistr.at<double>(y, x) * (double) remStat);
                    int32_t newval = nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) + val;
                    if (newval > statCorrsPRegNew.at<int32_t>(y, x)) {
                        int32_t diff = newval - statCorrsPRegNew.at<int32_t>(y, x);
                        val -= diff;
                        newval -= diff;
                        remStatrem -= val;
                        if (remStatrem < 0) {
                            val += remStatrem;
                            newval = nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) + val;
                            remStatrem = 0;
                        }
                        nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) = newval;
                    } else {
                        remStatrem -= val;
                        if (remStatrem < 0) {
                            nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) = newval + remStatrem;
                            remStatrem = 0;
                        } else {
                            nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) = newval;
                        }
                    }
                }
            }
        }
        if (remStatrem > 0) {
            vector<pair<int, int32_t>> availableCorrs(9);
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    const int idx = y * 3 + x;
                    availableCorrs[idx] = make_pair(idx, statCorrsPRegNew.at<int32_t>(y, x) -
                            nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x));
                }
            }
            sort(availableCorrs.begin(), availableCorrs.end(),
                 [](pair<int, int32_t> first, pair<int, int32_t> second) {
                     return first.second > second.second;
                 });
            int maxIt = remStatrem;
            while ((remStatrem > 0) && (maxIt > 0)) {
                for (int i = 0; i < 9; i++) {
                    if (availableCorrs[i].second > 0) {
                        int y = availableCorrs[i].first / 3;
                        int x = availableCorrs[i].first - y * 3;
                        nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x)++;
                        remStatrem--;
                        availableCorrs[i].second--;
                        if (remStatrem == 0) {
                            break;
                        }
                    }
                }
                maxIt--;
            }
        }
    }

    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if(verbose & PRINT_WARNING_MESSAGES) {
                if ((nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) == 0) && (statCorrsPRegNew.at<int32_t>(y, x) != 0)) {
                    cout << "Distributed correspondences on areas that should not hold correspondences!" << endl;
                }
            }
            nrTruePosRegs[actFrameCnt].at<int32_t>(y, x) = statCorrsPRegNew.at<int32_t>(y, x) -
                    nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x);
            nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) = statCorrsPRegNew.at<int32_t>(y, x);
        }
    }

    //Check the inlier ratio
    if(verbose & PRINT_WARNING_MESSAGES) {
        double tps = (double) sum(nrTruePosRegs[actFrameCnt])[0];
        double nrCorrs1 = (double) sum(nrCorrsRegs[actFrameCnt])[0];
        double inlRatDiffSR = tps / (nrCorrs1 + DBL_EPSILON) - inlRat[actFrameCnt];
        double testVal = min(nrCorrs1 / 100.0, 1.0) * inlRatDiffSR / 300.0;
        if (!nearZero(testVal)) {
            cout << "Inlier ratio of static correspondences after changing the number of correspondences per region"
                    " because of moving objects differs "
                    "from global inlier ratio (0 - 1.0) by "
                 << inlRatDiffSR << endl;
        }
    }
}

void genStereoSequ::adaptNrStaticCorrsBasedOnMovCorrs(const cv::Mat &mask){

    if(actCorrsOnMovObjFromLast <= 0){
        return;
    }

    //Remove correspondences from backprojected moving objects if there are too many of them
    auto maxNrOldMovCorrs = (int32_t) round(pars.CorrMovObjPort * (double) nrCorrs[actFrameCnt]);
    if(actCorrsOnMovObjFromLast > maxNrOldMovCorrs){
        int32_t remSize = actCorrsOnMovObjFromLast - maxNrOldMovCorrs;
        //Remove them based on the ratio of the number of correspondences
        adaptNrBPMovObjCorrs(remSize);
    }

    //Get overlap of regions and the portion of correspondences that is covered by the moving objects
    vector<vector<double>> movObjOverlap(3, vector<double>(3, 0));
    movObjHasArea = vector<vector<bool>>(3, vector<bool>(3, false));
    vector<vector<int32_t>> movObjCorrsFromStatic(3, vector<int32_t>(3, 0));
    vector<vector<int32_t>> movObjCorrsFromStaticInv(3, vector<int32_t>(3, 0));
    int32_t absNrCorrsFromStatic = 0;
    Mat statCorrsPRegNew = Mat::zeros(3, 3, CV_32SC1);
    double oldMovObjAreaImgRat = (double) cv::countNonZero(mask) / (double) imgSize.area();
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            movObjOverlap[y][x] = (double) (cv::countNonZero(mask(regROIs[y][x]))) /
                                  (double) (regROIs[y][x].area());
            CV_Assert(movObjOverlap[y][x] >= 0);
            if (movObjOverlap[y][x] > 0.9) {
                movObjHasArea[y][x] = true;
                movObjCorrsFromStatic[y][x] = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x);
                movObjCorrsFromStaticInv[y][x] = 0;
                absNrCorrsFromStatic += movObjCorrsFromStatic[y][x];
            } else if (nearZero(movObjOverlap[y][x])) {
                statCorrsPRegNew.at<int32_t>(y, x) = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x);
                movObjCorrsFromStatic[y][x] = 0;
                movObjCorrsFromStaticInv[y][x] = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x);
            } else {
                movObjCorrsFromStatic[y][x] = (int32_t) round(
                        (double) nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) * movObjOverlap[y][x]);
                int32_t maxFromOld = actCorrsOnMovObjFromLast;
                if(!nearZero(100.0 * oldMovObjAreaImgRat)) {
                    maxFromOld = (int32_t) round(
                            (double) actCorrsOnMovObjFromLast * movObjOverlap[y][x] / oldMovObjAreaImgRat);
                }
                movObjCorrsFromStatic[y][x] =
                        movObjCorrsFromStatic[y][x] > maxFromOld ? maxFromOld : movObjCorrsFromStatic[y][x];
                movObjCorrsFromStaticInv[y][x] =
                        nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) - movObjCorrsFromStatic[y][x];
                absNrCorrsFromStatic += movObjCorrsFromStatic[y][x];
                statCorrsPRegNew.at<int32_t>(y, x) = movObjCorrsFromStaticInv[y][x];
            }
        }
    }

    //Distribute the remaining # of correspondences on the regions
    int corrDiff = absNrCorrsFromStatic - actCorrsOnMovObjFromLast;

    if (corrDiff < 0)//Remove additional static correspondences and add them to the moving objects
    {
        int32_t remStat = actCorrsOnMovObjFromLast - absNrCorrsFromStatic;
        distributeStatObjCorrsOnMovObj(remStat,
                absNrCorrsFromStatic,
                movObjCorrsFromStaticInv,
                statCorrsPRegNew);

    } else if (corrDiff > 0)//Distribute a part of the correspondences from moving objects over the static elements not covered by moving objects
    {
        int32_t remMov = absNrCorrsFromStatic - actCorrsOnMovObjFromLast;
        cv::Mat mask_tmp = (mask == 0);
        distributeMovObjCorrsOnStatObj(remMov,
                                       absNrCorrsFromStatic,
                                       mask_tmp,
                                       movObjCorrsFromStaticInv,
                                       movObjOverlap,
                                       statCorrsPRegNew);

    }

    if(verbose & PRINT_WARNING_MESSAGES) {
        int32_t nrCorrsDiff = (int32_t)sum(statCorrsPRegNew)[0] + actCorrsOnMovObjFromLast - (int32_t)nrCorrs[actFrameCnt];
        if (nrCorrsDiff != 0) {
            cout << "Number of total correspondences differs by " << nrCorrsDiff << endl;
        }
    }

    //Set new number of static correspondences
    adaptStatNrCorrsReg(statCorrsPRegNew);

    actCorrsOnMovObj = 0;
    actTruePosOnMovObj = 0;
    actTrueNegOnMovObj = 0;
    actTPPerMovObj.clear();
    actTNPerMovObj.clear();
    movObjLabels.clear();
    combMovObjLabels = cv::Mat::zeros(imgSize, CV_8UC1);
    movObjLabelsROIs.clear();
}

void genStereoSequ::distributeMovObjCorrsOnStatObj(int32_t remMov,
                                    int32_t absNrCorrsFromStatic,
                                    const cv::Mat &movObjMask,
                                    std::vector<std::vector<int32_t>> movObjCorrsFromStaticInv,
                                    std::vector<std::vector<double>> movObjOverlap,
                                    cv::Mat &statCorrsPRegNew){
    CV_Assert(remMov >= 0);
    int32_t actStatCorrs = (int32_t)nrCorrs[actFrameCnt] - absNrCorrsFromStatic;
    CV_Assert(actStatCorrs >= 0);
    int32_t remMovrem = remMov;
    vector<vector<int32_t>> cmaxreg(3, vector<int32_t>(3, 0));
    cv::Mat fracUseableTPperRegion_tmp;

    vector<double> combDepths(3);
    combDepths[0] = actDepthMid;
    combDepths[1] = actDepthFar + (maxFarDistMultiplier - 1.0) * actDepthFar / 2.0;
    combDepths[2] = actDepthNear + (actDepthMid - actDepthNear) / 2.0;

    getInterSecFracRegions(fracUseableTPperRegion_tmp,
                           actR,
                           actT,
                           combDepths,
                           movObjMask);
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (!movObjHasArea[y][x] && (remMovrem > 0)) {
                int32_t val = movObjCorrsFromStaticInv[y][x];
                if(actStatCorrs != 0) {
                    val = (int32_t) round(
                            (double) movObjCorrsFromStaticInv[y][x] / (double) actStatCorrs * (double) remMov);
                }
                int32_t newval = movObjCorrsFromStaticInv[y][x] + val;
                //Get the maximum # of correspondences per area using the minimum distance between keypoints
                if (nearZero(actFracUseableTPperRegion.at<double>(y, x))) {
                    cmaxreg[y][x] = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x);
                } else if (!nearZero(actFracUseableTPperRegion.at<double>(y, x) - 1.0)) {
                    cmaxreg[y][x] = (int32_t) (
                            (double) ((regROIs[y][x].width - 1) * (regROIs[y][x].height - 1)) *
                            (1.0 - movObjOverlap[y][x]) * fracUseableTPperRegion_tmp.at<double>(y, x) /
                            (enlargeKPDist * avgMaskingArea));
                } else {
                    cmaxreg[y][x] = (int32_t) (
                            (double) ((regROIs[y][x].width - 1) * (regROIs[y][x].height - 1)) *
                            (1.0 - movObjOverlap[y][x]) /
                            (enlargeKPDist * avgMaskingArea));
                }
                if (newval <= cmaxreg[y][x]) {
                    remMovrem -= val;
                    if (remMovrem < 0) {
                        val += remMovrem;
                        newval = movObjCorrsFromStaticInv[y][x] + val;
                        remMovrem = 0;
                    }
                    statCorrsPRegNew.at<int32_t>(y, x) = newval;
                    cmaxreg[y][x] -= newval;
                } else {
                    if (movObjCorrsFromStaticInv[y][x] < cmaxreg[y][x]) {
                        statCorrsPRegNew.at<int32_t>(y, x) = cmaxreg[y][x];
                        remMovrem -= cmaxreg[y][x] - movObjCorrsFromStaticInv[y][x];
                        if (remMovrem < 0) {
                            statCorrsPRegNew.at<int32_t>(y, x) += remMovrem;
                            remMovrem = 0;
                        }
                        cmaxreg[y][x] -= statCorrsPRegNew.at<int32_t>(y, x);
                    } else {
                        cmaxreg[y][x] = 0;
                    }
                }
            }
        }
    }
    if (remMovrem > 0) {
        vector<pair<int, int32_t>> movObjCorrsFromStaticInv_tmp(9);
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                const int idx = y * 3 + x;
                movObjCorrsFromStaticInv_tmp[idx] = make_pair(idx, cmaxreg[y][x]);
            }
        }
        sort(movObjCorrsFromStaticInv_tmp.begin(), movObjCorrsFromStaticInv_tmp.end(),
             [](pair<int, int32_t> first, pair<int, int32_t> second) {
                 return first.second > second.second;
             });
        int maxIt = remMovrem;
        while ((remMovrem > 0) && (maxIt > 0)) {
            for (int i = 0; i < 9; i++) {
                int y = movObjCorrsFromStaticInv_tmp[i].first / 3;
                int x = movObjCorrsFromStaticInv_tmp[i].first - y * 3;
                if ((movObjCorrsFromStaticInv_tmp[i].second > 0) && (statCorrsPRegNew.at<int32_t>(y, x) > 0)) {
                    statCorrsPRegNew.at<int32_t>(y, x)++;
                    remMovrem--;
                    movObjCorrsFromStaticInv_tmp[i].second--;
                    if (remMovrem == 0) {
                        break;
                    }
                }
            }
            maxIt--;
        }
    }
}

void genStereoSequ::distributeStatObjCorrsOnMovObj(int32_t remStat,
                                    int32_t absNrCorrsFromStatic,
                                    std::vector<std::vector<int32_t>> movObjCorrsFromStaticInv,
                                    cv::Mat &statCorrsPRegNew){
    CV_Assert(remStat >= 0);
    int32_t actStatCorrs = (int32_t)nrCorrs[actFrameCnt] - absNrCorrsFromStatic;
    CV_Assert(actStatCorrs >= 0);
    if(actStatCorrs == 0)
        return;
    int32_t remStatrem = remStat;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (!movObjHasArea[y][x] && (remStatrem > 0)) {
                auto val = (int32_t) round(
                        (double) movObjCorrsFromStaticInv[y][x] / (double) actStatCorrs * (double) remStat);
                int32_t newval = movObjCorrsFromStaticInv[y][x] - val;
                if (newval > 0) {
                    remStatrem -= val;
                    if (remStatrem < 0) {
                        val += remStatrem;
                        newval = movObjCorrsFromStaticInv[y][x] - val;
                        remStatrem = 0;
                    }
                    statCorrsPRegNew.at<int32_t>(y, x) = newval;
                } else {
                    remStatrem -= val + newval;
                    if (remStatrem < 0) {
                        statCorrsPRegNew.at<int32_t>(y, x) = -remStatrem;
                        remStatrem = 0;
                    } else {
                        statCorrsPRegNew.at<int32_t>(y, x) = 0;
                    }
                }
            }
        }
    }
    if (remStatrem > 0) {
        vector<pair<int, int32_t>> movObjCorrsFromStaticInv_tmp(9);
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                const int idx = y * 3 + x;
                movObjCorrsFromStaticInv_tmp[idx] = make_pair(idx, statCorrsPRegNew.at<int32_t>(y, x));
            }
        }
        sort(movObjCorrsFromStaticInv_tmp.begin(), movObjCorrsFromStaticInv_tmp.end(),
             [](pair<int, int32_t> first, pair<int, int32_t> second) {
                 return first.second > second.second;
             });
        int maxIt = remStatrem;
        while ((remStatrem > 0) && (maxIt > 0)) {
            for (int i = 0; i < 9; i++) {
                if (movObjCorrsFromStaticInv_tmp[i].second > 0) {
                    int y = movObjCorrsFromStaticInv_tmp[i].first / 3;
                    int x = movObjCorrsFromStaticInv_tmp[i].first - y * 3;
                    statCorrsPRegNew.at<int32_t>(y, x)--;
                    remStatrem--;
                    movObjCorrsFromStaticInv_tmp[i].second--;
                    if (remStatrem == 0) {
                        break;
                    }
                }
            }
            maxIt--;
        }
    }
}

//Assign a depth category to each new object label and calculate all depth values for each label
void genStereoSequ::genNewDepthMovObj() {
    if (pars.nrMovObjs == 0)
        return;

    if(movObjLabels.empty()) {
        movObjDepthClassNew.clear();
        return;
    }

    //Get the depth classes that should be used for the new generated moving objects
    if (pars.movObjDepth.empty()) {
        pars.movObjDepth.push_back(depthClass::MID);
    }
    if (pars.movObjDepth.size() == pars.nrMovObjs)//Take for every moving object its corresponding depth
    {
        if (!movObjDepthClass.empty() && (movObjDepthClass.size() < pars.nrMovObjs)) {
            vector<bool> usedDepths(pars.movObjDepth.size(), false);
            for (size_t i = 0; i < pars.movObjDepth.size(); i++) {
                for (size_t j = 0; j < movObjDepthClass.size(); j++) {
                    if ((pars.movObjDepth[i] == movObjDepthClass[j]) && !usedDepths[i]) {
                        usedDepths[i] = true;
                        break;
                    }
                }
            }
            movObjDepthClassNew.resize(movObjLabels.size());
            for (size_t i = 0; i < movObjLabels.size(); i++) {
                for (size_t j = 0; j < usedDepths.size(); j++) {
                    if (!usedDepths[j]) {
                        usedDepths[j] = true;
                        movObjDepthClassNew[i] = pars.movObjDepth[j];
                        break;
                    }
                }
            }
        } else if (movObjDepthClass.empty()) {
            movObjDepthClassNew.clear();
            //copy(pars.movObjDepth.begin(), pars.movObjDepth.begin() + movObjLabels.size(), movObjDepthClassNew.begin());
            movObjDepthClassNew.insert(movObjDepthClassNew.end(), pars.movObjDepth.begin(),
                                       pars.movObjDepth.begin() + movObjLabels.size());
        } else {
            cout << "No new moving objects! This should not happen!" << endl;
            return;
        }
    } else if ((pars.movObjDepth.size() == 1))//Always take this depth class
    {
        movObjDepthClassNew.clear();
        movObjDepthClassNew.resize(movObjLabels.size(), pars.movObjDepth[0]);
    } else if ((pars.movObjDepth.size() < pars.nrMovObjs) && (pars.movObjDepth.size() > 1) &&
               (pars.movObjDepth.size() < 4))//Randomly choose a depth for every single object
    {
        movObjDepthClassNew.clear();
        movObjDepthClassNew.resize(movObjLabels.size());
        for (size_t i = 0; i < movObjLabels.size(); i++) {
            int rval = (int)(rand2() % pars.movObjDepth.size());
            movObjDepthClassNew[i] = pars.movObjDepth[rval];
        }
    } else//Use the given distribution of depth classes based on the number of given depth classes
    {
        movObjDepthClassNew.clear();
        movObjDepthClassNew.resize(movObjLabels.size());
        std::array<double, 3> depthDist = {{0, 0, 0}};
        for (auto& i : pars.movObjDepth) {
            switch (i) {
                case depthClass::NEAR:
                    depthDist[0]++;
                    break;
                case depthClass::MID:
                    depthDist[1]++;
                    break;
                case depthClass::FAR:
                    depthDist[2]++;
                    break;
                default:
                    break;
            }
        }
        std::discrete_distribution<int> distribution(depthDist.begin(), depthDist.end());
        for (size_t i = 0; i < movObjLabels.size(); i++) {
            int usedDepthClass = distribution(rand_gen);
            switch (usedDepthClass) {
                case 0:
                    movObjDepthClassNew[i] = depthClass::NEAR;
                    break;
                case 1:
                    movObjDepthClassNew[i] = depthClass::MID;
                    break;
                case 2:
                    movObjDepthClassNew[i] = depthClass::FAR;
                    break;
                default:
                    break;
            }
        }
    }

    //Get random parameters for the depth function
    std::vector<std::vector<double>> depthFuncPars;
    getRandDepthFuncPars(depthFuncPars, movObjDepthClassNew.size());

    //Get depth values for every pixel position inside the new labels
    combMovObjDepths = Mat::zeros(imgSize, CV_64FC1);
    for (size_t i = 0; i < movObjDepthClassNew.size(); i++) {
        double dmin = actDepthNear, dmax = actDepthMid;
        switch (movObjDepthClassNew[i]) {
            case depthClass::NEAR:
                dmin = actDepthNear;
                dmax = actDepthMid;
                break;
            case depthClass::MID:
                dmin = actDepthMid;
                dmax = actDepthFar;
                break;
            case depthClass::FAR:
                dmin = actDepthFar;
                dmax = maxFarDistMultiplier * actDepthFar;
                break;
            default:
                break;
        }

        double dmin_tmp = getRandDoubleValRng(dmin, dmin + 0.6 * (dmax - dmin));
        double dmax_tmp = getRandDoubleValRng(dmin_tmp + 0.1 * (dmax - dmin), dmax);
        double drange = dmax_tmp - dmin_tmp;
        double rXr = getRandDoubleValRng(1.5, 3.0);
        double rYr = getRandDoubleValRng(1.5, 3.0);
        auto h2 = (double) movObjLabelsROIs[i].height;
        h2 *= h2;
        auto w2 = (double) movObjLabelsROIs[i].width;
        w2 *= w2;
        double scale = sqrt(h2 + w2) / 2.0;
        double rXrSc = rXr / scale;
        double rYrSc = rYr / scale;
        double cx = (double) movObjLabelsROIs[i].width / 2.0;
        double cy = (double) movObjLabelsROIs[i].height / 2.0;

        double minVal = DBL_MAX, maxVal = -DBL_MAX;
        Mat objArea = movObjLabels[i](movObjLabelsROIs[i]);
        Mat objAreaDepths = combMovObjDepths(movObjLabelsROIs[i]);
        for (int y = 0; y < movObjLabelsROIs[i].height; y++) {
            for (int x = 0; x < movObjLabelsROIs[i].width; x++) {
                if (objArea.at<unsigned char>(y, x) != 0) {
                    double val = getDepthFuncVal(depthFuncPars[i], ((double) x - cx) * rXrSc,
                                                 ((double) y - cy) * rYrSc);
                    objAreaDepths.at<double>(y, x) = val;
                    if (val > maxVal)
                        maxVal = val;
                    if (val < minVal)
                        minVal = val;
                }
            }
        }
        double ra = maxVal - minVal;
        if(nearZero(ra))
            ra = 1.0;
        scale = drange / ra;
        for (int y = 0; y < movObjLabelsROIs[i].height; y++) {
            for (int x = 0; x < movObjLabelsROIs[i].width; x++) {
                if (objArea.at<unsigned char>(y, x) != 0) {
                    double val = objAreaDepths.at<double>(y, x);
                    val -= minVal;
                    val *= scale;
                    val += dmin_tmp;
                    objAreaDepths.at<double>(y, x) = val;
                }
            }
        }
    }

    //Visualize the depth values
    if (verbose & SHOW_MOV_OBJ_DISTANCES) {
        Mat normalizedDepth, labelMask = cv::Mat::zeros(imgSize, CV_8UC1);
        for (auto& dl : movObjLabels) {
            labelMask |= (dl != 0);
        }
        cv::normalize(combMovObjDepths, normalizedDepth, 0.1, 1.0, cv::NORM_MINMAX, -1, labelMask);
        if(!writeIntermediateImg(normalizedDepth, "normalized_moving_obj_depth")){
            namedWindow("Normalized Moving Obj Depth", WINDOW_AUTOSIZE);
            imshow("Normalized Moving Obj Depth", normalizedDepth);
            waitKey(0);
            destroyWindow("Normalized Moving Obj Depth");
        }
    }
}

void genStereoSequ::clearNewMovObjVars() {
    movObjCorrsImg1TP.clear();
    movObjCorrsImg2TP.clear();
    movObjCorrsImg1TN.clear();
    movObjCorrsImg2TN.clear();
    movObj3DPtsCamNew.clear();
    movObjDistTNtoRealNew.clear();
    actCorrsOnMovObj_IdxWorld.clear();
}

//Generate correspondences and TN for newly generated moving objects
void genStereoSequ::getMovObjCorrs() {
    size_t nr_movObj = actTPPerMovObj.size();
    if(nr_movObj == 0) {
        clearNewMovObjVars();
        movObjMaskFromLast2.copyTo(movObjMask2All);
        return;
    }
    int32_t kSi = csurr.rows;
    int32_t posadd = (kSi - 1) / 2;
    Point_<int32_t> pt;
    Point2d pt2;
    Point3d pCam;
    Mat corrsSet = Mat::zeros(imgSize.height + kSi - 1, imgSize.width + kSi - 1, CV_8UC1);
    cv::copyMakeBorder(movObjMaskFromLast2, movObjMask2All, posadd, posadd, posadd, posadd, BORDER_CONSTANT,
                       Scalar(0));//movObjMask2All must be reduced to image size at the end
    //int maxIt = 20;

    //For visualization
    int dispit = 0;
    const int dispit_interval = 50;

    //Generate TP correspondences
    clearNewMovObjVars();
    movObjCorrsImg1TP.resize(nr_movObj);
    movObjCorrsImg2TP.resize(nr_movObj);
    movObjCorrsImg1TN.resize(nr_movObj);
    movObjCorrsImg2TN.resize(nr_movObj);
    movObj3DPtsCamNew.resize(nr_movObj);
    movObjDistTNtoRealNew.resize(nr_movObj);
    actCorrsOnMovObj_IdxWorld.resize(nr_movObj);
    for (int i = 0; i < (int)nr_movObj; i++) {
        std::uniform_int_distribution<int32_t> distributionX(movObjLabelsROIs[i].x,
                                                             movObjLabelsROIs[i].x + movObjLabelsROIs[i].width - 1);
        std::uniform_int_distribution<int32_t> distributionY(movObjLabelsROIs[i].y,
                                                             movObjLabelsROIs[i].y + movObjLabelsROIs[i].height -
                                                             1);
        //int cnt1 = 0;
        int nrTN = actTNPerMovObj[i];
        int nrTP = actTPPerMovObj[i];
        vector<Point2d> x1TN, x2TN;
        vector<Point2d> x1TP, x2TP;
        int32_t maxSelect = 50;
        int32_t maxSelect2 = 50;
        int32_t maxSelect3 = 50;

        while ((nrTP > 0) && (maxSelect > 0) && (maxSelect2 > 0) && (maxSelect3 > 0)) {
            pt.x = distributionX(rand_gen);
            pt.y = distributionY(rand_gen);

            Mat s_tmp = corrsSet(Rect(pt, Size(kSi, kSi)));
            if ((movObjLabels[i].at<unsigned char>(pt) == 0) || (s_tmp.at<unsigned char>(posadd, posadd) > 0)) {
                maxSelect--;
                continue;
            }
            maxSelect = 50;

            //Check if it is also an inlier in the right image
            bool isInl = checkLKPInlier(pt, pt2, pCam, combMovObjDepths);
            if (isInl) {
                Mat s_tmp1 = movObjMask2All(Rect((int) round(pt2.x), (int) round(pt2.y), kSi, kSi));
                if (s_tmp1.at<unsigned char>(posadd, posadd) > 0) {
                    maxSelect2--;
                    continue;
                }
                s_tmp1.at<unsigned char>(posadd,
                                         posadd) = 1;//The minimum distance between keypoints in the second image is fixed to approx. 1 for new correspondences
                maxSelect2 = 50;
            }
            s_tmp += csurr;
            if (!isInl) {
                if (nrTN > 0) {
                    x1TN.emplace_back(Point2d((double) pt.x, (double) pt.y));
                    nrTN--;
                } else {
                    maxSelect3--;
                    s_tmp -= csurr;
                }
                continue;
            }
            maxSelect3 = 50;
            nrTP--;
            x1TP.emplace_back(Point2d((double) pt.x, (double) pt.y));
            x2TP.push_back(pt2);
            movObj3DPtsCamNew[i].push_back(pCam);

            //Visualize the masks
            if (verbose & SHOW_MOV_OBJ_CORRS_GEN) {
                if (dispit % dispit_interval == 0) {
                    if(!writeIntermediateImg((corrsSet > 0), "moving_obj_corrs_mask_img1_step_" + std::to_string(dispit)) ||
                       !writeIntermediateImg((movObjMask2All > 0), "moving_obj_corrs_mask_img2_step_" + std::to_string(dispit))){
                        namedWindow("Move Corrs mask img1", WINDOW_AUTOSIZE);
                        imshow("Move Corrs mask img1", (corrsSet > 0));
                        namedWindow("Move Corrs mask img2", WINDOW_AUTOSIZE);
                        imshow("Move Corrs mask img2", (movObjMask2All > 0));
                        waitKey(0);
                        destroyWindow("Move Corrs mask img1");
                        destroyWindow("Move Corrs mask img2");
                    }
                }
                dispit++;
            }
        }
        actCorrsOnMovObj_IdxWorld[i].resize(movObj3DPtsCamNew[i].size());
        std::iota(actCorrsOnMovObj_IdxWorld[i].begin(), actCorrsOnMovObj_IdxWorld[i].end(), 0);
        //If there are still correspondences missing, try to use them in the next object
        /*if ((nrTP > 0) && (i < (nr_movObj - 1)))
        {
            actTPPerMovObj[i + 1] += nrTP;
        }*/

        std::uniform_int_distribution<int32_t> distributionX2(0, imgSize.width - 1);
        std::uniform_int_distribution<int32_t> distributionY2(0, imgSize.height - 1);
        //Find TN keypoints in the second image for already found TN in the first image
        size_t corrsNotVisible = x1TN.size();
        if (!x1TN.empty()) {
            //Generate mask for visualization before adding keypoints
            Mat dispMask;
            if (verbose & SHOW_MOV_OBJ_CORRS_GEN) {
                dispMask = (movObjMask2All > 0);
            }

            for (size_t j = 0; j < corrsNotVisible; j++) {
                int max_try = 10;
                while (max_try > 0) {
                    pt.x = distributionX2(rand_gen);
                    pt.y = distributionY2(rand_gen);
                    Mat s_tmp = movObjMask2All(Rect(pt, Size(kSi, kSi)));
                    if (s_tmp.at<unsigned char>(posadd, posadd) > 0) {
                        max_try--;
                        continue;
                    }
                    //csurr.copyTo(s_tmp);
                    s_tmp.at<unsigned char>(posadd, posadd) = 1;
                    x2TN.emplace_back(Point2d((double) pt.x, (double) pt.y));
                    movObjDistTNtoRealNew[i].push_back(fakeDistTNCorrespondences);
                    break;
                }
            }
            while (x1TN.size() > x2TN.size()) {
                Mat s_tmp = corrsSet(Rect(Point_<int32_t>((int32_t) round(x1TN.back().x),
                                                          (int32_t) round(x1TN.back().y)), Size(kSi, kSi)));
                s_tmp -= csurr;
                x1TN.pop_back();
                nrTN++;
            }

            //Visualize the mask afterwards
            if (verbose & SHOW_MOV_OBJ_CORRS_GEN) {
                Mat dispMask2 = (movObjMask2All > 0);
                vector<Mat> channels;
                Mat b = Mat::zeros(dispMask2.size(), CV_8UC1);
                channels.push_back(b);
                channels.push_back(dispMask);
                channels.push_back(dispMask2);
                Mat img3c;
                merge(channels, img3c);
                if(!writeIntermediateImg(img3c, "moving_obj_TN_corrs_mask_img2")){
                    namedWindow("Move TN Corrs mask img2", WINDOW_AUTOSIZE);
                    imshow("Move TN Corrs mask img2", img3c);
                    waitKey(0);
                    destroyWindow("Move TN Corrs mask img2");
                }
            }
        }

        //Get the rest of TN correspondences
        if (nrTN > 0) {
            std::vector<Point2d> x1TN_tmp, x2TN_tmp;
            std::vector<double> x2TNdistCorr_tmp;
            Mat maskImg1;
            copyMakeBorder(movObjLabels[i], maskImg1, posadd, posadd, posadd, posadd, BORDER_CONSTANT, Scalar(0));
            maskImg1 = (maskImg1 == 0) | corrsSet;

            //Generate mask for visualization before adding keypoints
            Mat dispMaskImg2;
            Mat dispMaskImg1;
            if (verbose & SHOW_MOV_OBJ_CORRS_GEN) {
                dispMaskImg2 = (movObjMask2All > 0);
                dispMaskImg1 = (maskImg1 > 0);
            }

            //Generate a depth map for generating TN based on the depth of the actual moving object
            double dmin = actDepthNear, dmax = actDepthMid;
            switch (movObjDepthClassNew[i]) {
                case depthClass::NEAR:
                    dmin = actDepthNear;
                    dmax = actDepthMid;
                    break;
                case depthClass::MID:
                    dmin = actDepthMid;
                    dmax = actDepthFar;
                    break;
                case depthClass::FAR:
                    dmin = actDepthFar;
                    dmax = maxFarDistMultiplier * actDepthFar;
                    break;
                default:
                    break;
            }
            Mat randDepth(imgSize, CV_64FC1);
            randu(randDepth, Scalar(dmin), Scalar(dmax + 0.001));

            nrTN = genTrueNegCorrs(nrTN, distributionX, distributionY, distributionX2, distributionY2, x1TN_tmp,
                                   x2TN_tmp, x2TNdistCorr_tmp, maskImg1, movObjMask2All,
                                   randDepth);//, movObjLabels[i]);

            //Visualize the mask afterwards
            if (verbose & SHOW_MOV_OBJ_CORRS_GEN) {
                Mat dispMask2Img2 = (movObjMask2All > 0);
                Mat dispMask2Img1 = (maskImg1 > 0);
                vector<Mat> channels, channels1;
                Mat b = Mat::zeros(dispMask2Img2.size(), CV_8UC1);
                channels.push_back(b);
                channels.push_back(dispMaskImg2);
                channels.push_back(dispMask2Img2);
                channels1.push_back(b);
                channels1.push_back(dispMaskImg1);
                channels1.push_back(dispMask2Img1);
                Mat img3c, img3c1;
                merge(channels, img3c);
                merge(channels1, img3c1);
                if(!writeIntermediateImg(img3c1, "moving_obj_rand_TN_corrs_mask_img1") ||
                   !writeIntermediateImg(img3c, "moving_obj_rand_TN_corrs_mask_img2")){
                    namedWindow("Move rand TN Corrs mask img1", WINDOW_AUTOSIZE);
                    imshow("Move rand TN Corrs mask img1", img3c1);
                    namedWindow("Move rand TN Corrs mask img2", WINDOW_AUTOSIZE);
                    imshow("Move rand TN Corrs mask img2", img3c);
                    waitKey(0);
                    destroyWindow("Move rand TN Corrs mask img1");
                    destroyWindow("Move rand TN Corrs mask img2");
                }
            }

            if (!x1TN_tmp.empty()) {
                corrsSet(Rect(Point(posadd, posadd), imgSize)) |= (maskImg1(Rect(Point(posadd, posadd), imgSize)) &
                                                                   (movObjLabels[i] > 0));
                //copy(x1TN_tmp.begin(), x1TN_tmp.end(), x1TN.end());
                x1TN.insert(x1TN.end(), x1TN_tmp.begin(), x1TN_tmp.end());
                //copy(x2TN_tmp.begin(), x2TN_tmp.end(), x2TN.end());
                x2TN.insert(x2TN.end(), x2TN_tmp.begin(), x2TN_tmp.end());
                //copy(x2TNdistCorr_tmp.begin(), x2TNdistCorr_tmp.end(), movObjDistTNtoRealNew[i].end());
                movObjDistTNtoRealNew[i].insert(movObjDistTNtoRealNew[i].end(), x2TNdistCorr_tmp.begin(),
                                                x2TNdistCorr_tmp.end());
            }
        }

        //Adapt the number of TP and TN in the next objects based on the remaining number of TP and TN of the current object
        adaptNRCorrespondences(nrTP, nrTN, corrsNotVisible, x1TP.size(), i, (int32_t)nr_movObj);

        //Store correspondences
        if (!x1TP.empty()) {
            movObjCorrsImg1TP[i] = Mat::ones(3, (int) x1TP.size(), CV_64FC1);
            movObjCorrsImg2TP[i] = Mat::ones(3, (int) x1TP.size(), CV_64FC1);
            movObjCorrsImg1TP[i].rowRange(0, 2) = Mat(x1TP).reshape(1).t();
            movObjCorrsImg2TP[i].rowRange(0, 2) = Mat(x2TP).reshape(1).t();
        }
        if (!x1TN.empty()) {
            movObjCorrsImg1TN[i] = Mat::ones(3, (int) x1TN.size(), CV_64FC1);
            movObjCorrsImg1TN[i].rowRange(0, 2) = Mat(x1TN).reshape(1).t();
        }
        if (!x2TN.empty()) {
            movObjCorrsImg2TN[i] = Mat::ones(3, (int) x2TN.size(), CV_64FC1);
            movObjCorrsImg2TN[i].rowRange(0, 2) = Mat(x2TN).reshape(1).t();
        }
    }
    movObjMask2All = movObjMask2All(Rect(Point(posadd, posadd), imgSize));

    //Check number of TP and TN per moving object and the overall inlier ratio
    if(verbose & PRINT_WARNING_MESSAGES) {
        int nrCorrsMO = 0, nrTPMO = 0, nrTNMO = 0;
        vector<int> nrTPperMO(nr_movObj, 0), nrTNperMO(nr_movObj, 0);
        for (size_t k = 0; k < nr_movObj; ++k) {
            nrTPperMO[k] = movObjCorrsImg1TP[k].cols;
            nrTPMO += nrTPperMO[k];
            nrTNperMO[k] = movObjCorrsImg1TN[k].cols;
            nrTNMO += nrTNperMO[k];
        }
        nrCorrsMO = nrTPMO + nrTNMO;
        if (nrCorrsMO != actCorrsOnMovObj) {
            double chRate = (double) nrCorrsMO / (double) actCorrsOnMovObj;
            if ((chRate < 0.90) || (chRate > 1.10)) {
                cout << "Number of correspondences on moving objects is " << 100.0 * (chRate - 1.0)
                     << "% different to given values!" << endl;
                cout << "Actual #: " << nrCorrsMO << " Given #: " << actCorrsOnMovObj << endl;
                for (size_t k = 0; k < nr_movObj; ++k) {
                    if ((int32_t) nrTPperMO[k] != actTPPerMovObj[k]) {
                        cout << "# of TP for moving object " << k << " at position (x, y): (" <<
                             movObjLabelsROIs[k].x + movObjLabelsROIs[k].width / 2 <<
                             ", " << movObjLabelsROIs[k].y +
                                     movObjLabelsROIs[k].height / 2
                             << ") differs by "
                             << (int32_t) nrTPperMO[k] - actTPPerMovObj[k]
                             <<
                             " correspondences (Actual #: " << nrTPperMO[k]
                             << " Given #: " << actTPPerMovObj[k] << ")"
                             << endl;
                    }
                    if ((int32_t) nrTNperMO[k] != actTNPerMovObj[k]) {
                        cout << "# of TN for moving object " << k << " at position (x, y): (" <<
                             movObjLabelsROIs[k].x + movObjLabelsROIs[k].width / 2 <<
                             ", " << movObjLabelsROIs[k].y +
                                     movObjLabelsROIs[k].height / 2
                             << ") differs by "
                             << (int32_t) nrTNperMO[k] - actTNPerMovObj[k]
                             <<
                             " correspondences (Actual #: " << nrTNperMO[k]
                             << " Given #: " << actTNPerMovObj[k] << ")"
                             << endl;
                    }
                }
            }
        }
        double inlRatDiffMO = (double) nrTPMO / (double) nrCorrsMO - inlRat[actFrameCnt];
        double testVal = min((double)nrCorrsMO / 100.0, 1.0) * inlRatDiffMO / 300.0;
        if (!nearZero(testVal)) {
            cout << "Inlier ratio of moving object correspondences differs from global inlier ratio (0 - 1.0) by "
                 << inlRatDiffMO << endl;
        }
    }

    //Remove empty moving object point clouds
    vector<int> delList;
    for (int l = 0; l < (int)movObj3DPtsCamNew.size(); ++l) {
        if(movObj3DPtsCamNew[l].empty()){
            delList.push_back(l);
        }
    }
    if(!delList.empty()){
        for (int i = (int)delList.size() - 1; i >= 0; --i) {
            movObj3DPtsCamNew.erase(movObj3DPtsCamNew.begin() + delList[i]);
            movObjDepthClassNew.erase(movObjDepthClassNew.begin() + delList[i]);
            actCorrsOnMovObj_IdxWorld.erase(actCorrsOnMovObj_IdxWorld.begin() + delList[i]);
        }
    }
    /*for (auto itr = movObj3DPtsCamNew.rbegin();
         itr != movObj3DPtsCamNew.rend(); itr++) {
        if (itr->empty()) {
            movObj3DPtsCamNew.erase(std::next(itr).base());
        }
    }*/
}

//Generate (backproject) correspondences from existing moving objects and generate hulls of the objects in the image
//Moreover, as many true negatives as needed by the inlier ratio are generated
void genStereoSequ::backProjectMovObj() {
    convhullPtsObj.clear();
    actTNPerMovObjFromLast.clear();
    movObjLabelsFromLast.clear();
    movObjCorrsImg1TPFromLast.clear();
    movObjCorrsImg2TPFromLast.clear();
    movObjCorrsImg12TPFromLast_Idx.clear();
    movObjCorrsImg1TNFromLast.clear();
    movObjCorrsImg2TNFromLast.clear();
    movObjDistTNtoReal.clear();
    movObjMaskFromLast = Mat::zeros(imgSize, CV_8UC1);
    movObjMaskFromLast2 = Mat::zeros(imgSize, CV_8UC1);
    actCorrsOnMovObjFromLast = 0;
    actTruePosOnMovObjFromLast = 0;
    actTrueNegOnMovObjFromLast = 0;

    if (pars.nrMovObjs == 0) {
        return;
    }

    vector<size_t> delList;
    size_t actNrMovObj = movObj3DPtsCam.size();

    if (movObj3DPtsCam.empty())
        return;

    movObjCorrsImg1TPFromLast.resize(actNrMovObj);
    movObjCorrsImg2TPFromLast.resize(actNrMovObj);
    movObjCorrsImg12TPFromLast_Idx.resize(actNrMovObj);
    movObjCorrsImg1TNFromLast.resize(actNrMovObj);
    movObjCorrsImg2TNFromLast.resize(actNrMovObj);

    struct imgWH {
        double width;
        double height;
        double maxDist;
    } dimgWH = {0, 0, 0};
    dimgWH.width = (double) (imgSize.width - 1);
    dimgWH.height = (double) (imgSize.height - 1);
    dimgWH.maxDist = maxFarDistMultiplier * actDepthFar;
    int sqrSi = csurr.rows;
    int posadd = (sqrSi - 1) / 2;
    std::vector<Mat> movObjMaskFromLastLarge(actNrMovObj);
    std::vector<Mat> movObjMaskFromLastLarge2(actNrMovObj);
    movObjMaskFromLastLargeAdd = std::vector<Mat>(actNrMovObj);
    std::vector<std::vector<cv::Point>> movObjPt1(actNrMovObj), movObjPt2(actNrMovObj);

    //Get correspondences (TN + TP) of backprojected moving objects
    for (size_t i = 0; i < actNrMovObj; i++) {
        movObjMaskFromLastLarge[i] = Mat::zeros(imgSize.height + sqrSi - 1, imgSize.width + sqrSi - 1, CV_8UC1);
        movObjMaskFromLastLarge2[i] = Mat::zeros(imgSize.height + sqrSi - 1, imgSize.width + sqrSi - 1, CV_8UC1);
        size_t oor = 0;
        size_t i2 = 0;
        for (const auto& pt : movObj3DPtsCam[i]) {
            i2++;
            if ((pt.z < actDepthNear) ||
                (pt.z > dimgWH.maxDist)) {
                oor++;
                continue;
            }

            Mat X = Mat(pt, false).reshape(1, 3);
            Mat x1 = K1 * X;
            if(nearZero(x1.at<double>(2))){
                oor++;
                continue;
            }
            x1 /= x1.at<double>(2);

            bool outOfR[2] = {false, false};
            if ((x1.at<double>(0) < 0) || (x1.at<double>(0) > dimgWH.width) ||
                (x1.at<double>(1) < 0) || (x1.at<double>(1) > dimgWH.height))//Not visible in first image
            {
                outOfR[0] = true;
            }

            Mat x2 = K2 * (actR * X + actT);
            if(nearZero(x2.at<double>(2))){
                oor++;
                continue;
            }
            x2 /= x2.at<double>(2);

            if ((x2.at<double>(0) < 0) || (x2.at<double>(0) > dimgWH.width) ||
                (x2.at<double>(1) < 0) || (x2.at<double>(1) > dimgWH.height))//Not visible in second image
            {
                outOfR[1] = true;
            }

            if (outOfR[0] || outOfR[1])
                oor++;

            Point ptr1 = Point((int) round(x1.at<double>(0)), (int) round(x1.at<double>(1)));
            Point ptr2 = Point((int) round(x2.at<double>(0)), (int) round(x2.at<double>(1)));

            //Check if the point is too near to an other correspondence of a moving object
            if (!outOfR[0] && movObjMaskFromLast.at<unsigned char>(ptr1) > 0)
                outOfR[0] = true;

            //Check if the point is too near to an other correspondence of a moving object in the second image
            if (!outOfR[1] && movObjMaskFromLast2.at<unsigned char>(ptr2) > 0)
                outOfR[1] = true;

            //Check if the point is too near to an other correspondence of this moving object
            if (!outOfR[0]) {
                Mat s_tmp = movObjMaskFromLastLarge[i](Rect(ptr1, Size(sqrSi, sqrSi)));
                if (s_tmp.at<unsigned char>(posadd, posadd) > 0)
                    outOfR[0] = true;
                else {
//                    csurr.copyTo(s_tmp);
                    s_tmp += csurr;
                }
            }
            //Check if the point is too near to an other correspondence of this moving object in the second image
            if (!outOfR[1]) {
                Mat s_tmp = movObjMaskFromLastLarge2[i](Rect(ptr2, Size(sqrSi, sqrSi)));
                if (s_tmp.at<unsigned char>(posadd, posadd) > 0)
                    outOfR[1] = true;
                else {
//                    csurr.copyTo(s_tmp);
                    s_tmp.at<unsigned char>(posadd, posadd) = 1;
                }
            }

            if (outOfR[0] && outOfR[1]) {
                continue;
            } else if (outOfR[0]) {
                if (movObjCorrsImg2TNFromLast[i].empty()) {
                    movObjCorrsImg2TNFromLast[i] = x2.t();
                } else {
                    movObjCorrsImg2TNFromLast[i].push_back(x2.t());
                }
                movObjPt2[i].push_back(ptr2);
            } else if (outOfR[1]) {
                if (movObjCorrsImg1TNFromLast[i].empty()) {
                    movObjCorrsImg1TNFromLast[i] = x1.t();
                } else {
                    movObjCorrsImg1TNFromLast[i].push_back(x1.t());
                }
                movObjPt1[i].push_back(ptr1);
            } else {
                if (movObjCorrsImg1TPFromLast[i].empty()) {
                    movObjCorrsImg1TPFromLast[i] = x1.t();
                    movObjCorrsImg2TPFromLast[i] = x2.t();
                } else {
                    movObjCorrsImg1TPFromLast[i].push_back(x1.t());
                    movObjCorrsImg2TPFromLast[i].push_back(x2.t());
                }
                movObjCorrsImg12TPFromLast_Idx[i].push_back(i2 - 1);
                movObjPt1[i].push_back(ptr1);
                movObjPt2[i].push_back(ptr2);
            }
        }

        movObjMaskFromLastLarge[i].copyTo(movObjMaskFromLastLargeAdd[i]);
        movObjMaskFromLastLarge[i] = (movObjMaskFromLastLarge[i] > 0) & Mat::ones(imgSize.height + sqrSi - 1,
                imgSize.width + sqrSi - 1, CV_8UC1);

        //Check if the portion of usable 3D points of this moving  object is below a user specified threshold. If yes, delete it.
        double actGoodPortion = 0;
        if (!movObj3DPtsCam[i].empty()) {
            actGoodPortion = (double) (movObj3DPtsCam[i].size() - oor) / (double) movObj3DPtsCam[i].size();
            if(movObjCorrsImg12TPFromLast_Idx[i].empty()){
                actGoodPortion = 0;
            }
        }
        if ((actGoodPortion < pars.minMovObjCorrPortion) || nearZero(actGoodPortion)) {
            delList.push_back(i);
        } else {
            if (!movObjCorrsImg1TNFromLast[i].empty())
                movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].t();
            if (!movObjCorrsImg2TNFromLast[i].empty())
                movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].t();
            if (!movObjCorrsImg1TPFromLast[i].empty()) {
                movObjCorrsImg1TPFromLast[i] = movObjCorrsImg1TPFromLast[i].t();
                movObjCorrsImg2TPFromLast[i] = movObjCorrsImg2TPFromLast[i].t();
            }

            Mat dispMask1, dispMask2;
            if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                dispMask1 = (movObjMaskFromLast > 0);
                dispMask2 = (movObjMaskFromLast2 > 0);
            }

            movObjMaskFromLast |= movObjMaskFromLastLarge[i](Rect(Point(posadd, posadd), imgSize));
            movObjMaskFromLast2 |= movObjMaskFromLastLarge2[i](Rect(Point(posadd, posadd), imgSize));

            if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                Mat dispMask12 = (movObjMaskFromLast > 0);
                Mat dispMask22 = (movObjMaskFromLast2 > 0);
                vector<Mat> channels;
                Mat b = Mat::zeros(dispMask12.size(), CV_8UC1);
                channels.push_back(b);
                channels.push_back(dispMask1);
                channels.push_back(dispMask12);
                Mat img3c;
                merge(channels, img3c);
                if(!writeIntermediateImg(img3c, "backprojected_moving_obj_mask_of_TP_and_TN_corrs_img1")){
                    namedWindow("Backprojected moving objects mask of TP and TN image 1", WINDOW_AUTOSIZE);
                    imshow("Backprojected moving objects mask of TP and TN image 1", img3c);
                }
                channels.clear();
                channels.push_back(b);
                channels.push_back(dispMask2);
                channels.push_back(dispMask22);
                merge(channels, img3c);
                if(!writeIntermediateImg(img3c, "backprojected_moving_obj_mask_of_TP_and_TN_corrs_img2")){
                    namedWindow("Backprojected moving objects mask of TP and TN image 2", WINDOW_AUTOSIZE);
                    imshow("Backprojected moving objects mask of TP and TN image 2", img3c);
                    waitKey(0);
                    destroyWindow("Backprojected moving objects mask of TP and TN image 1");
                    destroyWindow("Backprojected moving objects mask of TP and TN image 2");
                }
            }
        }
    }

    if (!delList.empty()) {
        for (int i = (int) delList.size() - 1; i >= 0; i--) {
            movObjCorrsImg1TNFromLast.erase(movObjCorrsImg1TNFromLast.begin() + delList[i]);
            movObjCorrsImg2TNFromLast.erase(movObjCorrsImg2TNFromLast.begin() + delList[i]);
            movObjCorrsImg1TPFromLast.erase(movObjCorrsImg1TPFromLast.begin() + delList[i]);
            movObjCorrsImg2TPFromLast.erase(movObjCorrsImg2TPFromLast.begin() + delList[i]);
            movObjCorrsImg12TPFromLast_Idx.erase(movObjCorrsImg12TPFromLast_Idx.begin() + delList[i]);
            movObjMaskFromLast &= (movObjMaskFromLastLarge[delList[i]](Rect(Point(posadd, posadd), imgSize)) == 0);
            movObjMaskFromLast2 &= (movObjMaskFromLastLarge2[delList[i]](Rect(Point(posadd, posadd), imgSize)) == 0);
            movObjMaskFromLastLarge.erase(movObjMaskFromLastLarge.begin() + delList[i]);
            movObjMaskFromLastLarge2.erase(movObjMaskFromLastLarge2.begin() + delList[i]);
            movObjMaskFromLastLargeAdd.erase(movObjMaskFromLastLargeAdd.begin() + delList[i]);
            movObjPt1.erase(movObjPt1.begin() + delList[i]);
            movObjPt2.erase(movObjPt2.begin() + delList[i]);

            movObj3DPtsCam.erase(movObj3DPtsCam.begin() + delList[i]);
            movObj3DPtsWorld.erase(movObj3DPtsWorld.begin() + delList[i]);
            movObjFrameEmerge.erase(movObjFrameEmerge.begin() + delList[i]);
            movObjImg12TP_InitIdxWorld.erase(movObjImg12TP_InitIdxWorld.begin() + delList[i]);
            actCorrsOnMovObjFromLast_IdxWorld.erase(actCorrsOnMovObjFromLast_IdxWorld.begin() + delList[i]);
            movObjWorldMovement.erase(movObjWorldMovement.begin() + delList[i]);
            movObjDepthClass.erase(movObjDepthClass.begin() + delList[i]);
        }
        actNrMovObj = movObj3DPtsCam.size();
        if (actNrMovObj == 0)
            return;
    }
    movObjDistTNtoReal.resize(actNrMovObj);

    //Remove TN if we have too much of them based on the used inlier ratio
    for (size_t i = 0; i < actNrMovObj; ++i) {
        auto maxTNthisMO =
                (int) round((double) movObjCorrsImg1TPFromLast[i].cols * (1.0 / inlRat[actFrameCnt] - 1.0));
        //Remove TN in the first image
        if(maxTNthisMO < movObjCorrsImg1TNFromLast[i].cols){
            Point pt;
            for (int j = maxTNthisMO; j < movObjCorrsImg1TNFromLast[i].cols; ++j) {
                pt = Point((int)round(movObjCorrsImg1TNFromLast[i].at<double>(0,j)),
                           (int)round(movObjCorrsImg1TNFromLast[i].at<double>(1,j)));
                Mat s_tmp = movObjMaskFromLastLargeAdd[i](Rect(pt, Size(sqrSi, sqrSi)));
                s_tmp -= csurr;
            }
            movObjMaskFromLast &= (movObjMaskFromLastLarge[i](Rect(Point(posadd, posadd), imgSize)) == 0);
            movObjMaskFromLastLarge[i] = (movObjMaskFromLastLargeAdd[i] > 0) & Mat::ones(imgSize.height + sqrSi - 1,
                                                                                      imgSize.width + sqrSi - 1, CV_8UC1);
            movObjMaskFromLast |= movObjMaskFromLastLarge[i](Rect(Point(posadd, posadd), imgSize));

            movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].colRange(0, maxTNthisMO);
        }
        //Remove TN in the second image
        if(maxTNthisMO < movObjCorrsImg2TNFromLast[i].cols){
            Point pt;
            for (int j = maxTNthisMO; j < movObjCorrsImg2TNFromLast[i].cols; ++j) {
                pt = Point((int)round(movObjCorrsImg2TNFromLast[i].at<double>(0,j)),
                           (int)round(movObjCorrsImg2TNFromLast[i].at<double>(1,j)));
                movObjMaskFromLast2.at<unsigned char>(pt) = 0;
                movObjMaskFromLastLarge2[i].at<unsigned char>(pt.y + posadd, pt.x + posadd) = 0;
            }
            movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].colRange(0, maxTNthisMO);
        }
    }

    //Generate hulls of the objects in the image and a mask fo every moving object and a global label mask
    movObjLabelsFromLast.resize(actNrMovObj);
    convhullPtsObj.resize(actNrMovObj);
    Mat movObjMaskFromLastOld = movObjMaskFromLast.clone();
    movObjMaskFromLast = Mat::zeros(imgSize, CV_8UC1);
    //vector<double> actAreaMovObj(actNrMovObj, 0);
    delList.clear();
    for (size_t i = 0; i < actNrMovObj; i++) {
        Mat movObjMaskFromLastLargePiece = movObjMaskFromLastLarge[i](Rect(Point(posadd, posadd), imgSize));
        genMovObjHulls(movObjMaskFromLastLargePiece, movObjPt1[i], movObjLabelsFromLast[i]);
        if (i > 0) {
            movObjLabelsFromLast[i] &= (movObjMaskFromLast == 0);
            int movObjLabelSizePix = countNonZero(movObjLabelsFromLast[i]);
            if(movObjLabelSizePix == 0){
                delList.push_back(i);
            }
        }

        Mat dispMask;
        if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
            dispMask = (movObjMaskFromLast > 0);
        }

        movObjMaskFromLast |= movObjLabelsFromLast[i];

        if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
            Mat dispMask2 = (movObjMaskFromLast > 0);
            vector<Mat> channels;
            Mat b = Mat::zeros(dispMask2.size(), CV_8UC1);
            channels.push_back(b);
            channels.push_back(dispMask);
            channels.push_back(dispMask2);
            Mat img3c;
            merge(channels, img3c);
            if(!writeIntermediateImg(img3c, "backprojected_moving_obj_hulls")){
                namedWindow("Backprojected moving object hulls", WINDOW_AUTOSIZE);
                imshow("Backprojected moving object hulls", img3c);
                waitKey(0);
                destroyWindow("Backprojected moving object hulls");
            }
        }
    }

    if(!delList.empty()){
        for (int i = (int) delList.size() - 1; i >= 0; i--) {
            movObjCorrsImg1TNFromLast.erase(movObjCorrsImg1TNFromLast.begin() + delList[i]);
            movObjCorrsImg2TNFromLast.erase(movObjCorrsImg2TNFromLast.begin() + delList[i]);
            movObjCorrsImg1TPFromLast.erase(movObjCorrsImg1TPFromLast.begin() + delList[i]);
            movObjCorrsImg2TPFromLast.erase(movObjCorrsImg2TPFromLast.begin() + delList[i]);
            movObjCorrsImg12TPFromLast_Idx.erase(movObjCorrsImg12TPFromLast_Idx.begin() + delList[i]);
            movObjMaskFromLastOld &= (movObjMaskFromLastLarge[delList[i]](Rect(Point(posadd, posadd), imgSize)) == 0);
            movObjMaskFromLast2 &= (movObjMaskFromLastLarge2[delList[i]](Rect(Point(posadd, posadd), imgSize)) == 0);
            movObjMaskFromLastLarge.erase(movObjMaskFromLastLarge.begin() + delList[i]);
            movObjMaskFromLastLarge2.erase(movObjMaskFromLastLarge2.begin() + delList[i]);
            movObjMaskFromLastLargeAdd.erase(movObjMaskFromLastLargeAdd.begin() + delList[i]);
            movObjPt1.erase(movObjPt1.begin() + delList[i]);
            movObjPt2.erase(movObjPt2.begin() + delList[i]);

            movObj3DPtsCam.erase(movObj3DPtsCam.begin() + delList[i]);
            movObj3DPtsWorld.erase(movObj3DPtsWorld.begin() + delList[i]);
            movObjFrameEmerge.erase(movObjFrameEmerge.begin() + delList[i]);
            movObjImg12TP_InitIdxWorld.erase(movObjImg12TP_InitIdxWorld.begin() + delList[i]);
            actCorrsOnMovObjFromLast_IdxWorld.erase(actCorrsOnMovObjFromLast_IdxWorld.begin() + delList[i]);
            movObjWorldMovement.erase(movObjWorldMovement.begin() + delList[i]);
            movObjDepthClass.erase(movObjDepthClass.begin() + delList[i]);
            movObjDistTNtoReal.erase(movObjDistTNtoReal.begin() + delList[i]);

            movObjLabelsFromLast.erase(movObjLabelsFromLast.begin() + delList[i]);
            convhullPtsObj.erase(convhullPtsObj.begin() + delList[i]);
        }
        actNrMovObj = movObj3DPtsCam.size();
        if (actNrMovObj == 0)
            return;
    }

    //Enlarge the object areas if they are too small
    Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    const int maxCnt = 40;
    for (size_t i = 0; i < actNrMovObj; i++) {
        int areaMO = cv::countNonZero(movObjLabelsFromLast[i]);
        if(areaMO == 0){
            if(verbose & SHOW_IMGS_AT_ERROR) {
                if(!writeIntermediateImg(movObjMaskFromLast, "error_zero_backprojected_moving_obj_are_whole_mask_-_obj_nr_" + std::to_string(i)) ||
                   !writeIntermediateImg(movObjMaskFromLastLarge[i], "error_zero_backprojected_moving_obj_are_marked_keypoints_-_obj_nr_" + std::to_string(i))){
                    namedWindow("Error - Backprojected moving object area zero - whole mask", WINDOW_AUTOSIZE);
                    imshow("Error - Backprojected moving object area zero - whole mask", movObjMaskFromLast);
                    namedWindow("Error - Backprojected moving object area zero - marked keypoints", WINDOW_AUTOSIZE);
                    imshow("Error - Backprojected moving object area zero - marked keypoints", movObjMaskFromLastLarge[i]);
                    waitKey(0);
                    destroyWindow("Error - Backprojected moving object area zero - whole mask");
                    destroyWindow("Error - Backprojected moving object area zero - marked keypoints");
                }
            }
            throw SequenceException("Label area of backprojected moving object is zero!");
        }
        int cnt = 0;
        while ((areaMO < minOArea) && (cnt < maxCnt)) {
            Mat imgSDdilate;
            dilate(movObjLabelsFromLast[i], imgSDdilate, element);
            imgSDdilate &= ((movObjMaskFromLast == 0) | movObjLabelsFromLast[i]);
            int areaMO2 = cv::countNonZero(imgSDdilate);
            if (areaMO2 > areaMO) {
                areaMO = areaMO2;
                imgSDdilate.copyTo(movObjLabelsFromLast[i]);
                if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                    if ((cnt % 4) == 0) {
                        if(!writeIntermediateImg(movObjLabelsFromLast[i] > 0,
                                "backprojected_moving_obj_nr" + std::to_string(i) + "_hull_enlargement_step_" + std::to_string(cnt))){
                            namedWindow("Backprojected moving object hull enlargement", WINDOW_AUTOSIZE);
                            imshow("Backprojected moving object hull enlargement", movObjLabelsFromLast[i] > 0);
                            waitKey(0);
                            destroyWindow("Backprojected moving object hull enlargement");
                        }
                    }
                }
            } else {
                break;
            }
            cnt++;
        }
        if (cnt > 0) {
            Mat dispMask;
            if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                dispMask = (movObjMaskFromLast > 0);
            }

            movObjMaskFromLast |= movObjLabelsFromLast[i];

            if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                Mat dispMask2 = (movObjMaskFromLast > 0);
                vector<Mat> channels;
                Mat b = Mat::zeros(dispMask2.size(), CV_8UC1);
                channels.push_back(b);
                channels.push_back(dispMask);
                channels.push_back(dispMask2);
                Mat img3c;
                merge(channels, img3c);
                if(!writeIntermediateImg(img3c, "backprojected_dilated_moving_obj_hulls")){
                    namedWindow("Backprojected dilated moving object hulls", WINDOW_AUTOSIZE);
                    imshow("Backprojected dilated moving object hulls", img3c);
                    waitKey(0);
                    destroyWindow("Backprojected dilated moving object hulls");
                }
            }
        }
    }

    //Get missing TN correspondences in second image for found TN keypoints in first image using found TN keypoints in second image
    vector<int> missingCImg2(actNrMovObj, 0);
    vector<int> missingCImg1(actNrMovObj, 0);
    for (size_t i = 0; i < actNrMovObj; i++) {
        if (!movObjCorrsImg1TNFromLast[i].empty() && !movObjCorrsImg2TNFromLast[i].empty()) {
            if (movObjCorrsImg1TNFromLast[i].cols > movObjCorrsImg2TNFromLast[i].cols) {
                missingCImg2[i] = movObjCorrsImg1TNFromLast[i].cols - movObjCorrsImg2TNFromLast[i].cols;
                movObjDistTNtoReal[i] = vector<double>((size_t)movObjCorrsImg1TNFromLast[i].cols, fakeDistTNCorrespondences);
            } else if (movObjCorrsImg1TNFromLast[i].cols < movObjCorrsImg2TNFromLast[i].cols) {
                missingCImg1[i] = movObjCorrsImg2TNFromLast[i].cols - movObjCorrsImg1TNFromLast[i].cols;
                movObjDistTNtoReal[i] = vector<double>((size_t)movObjCorrsImg2TNFromLast[i].cols, fakeDistTNCorrespondences);
            } else {
                movObjDistTNtoReal[i] = vector<double>((size_t)movObjCorrsImg1TNFromLast[i].cols, fakeDistTNCorrespondences);
            }
        } else if (!movObjCorrsImg1TNFromLast[i].empty()) {
            missingCImg2[i] = movObjCorrsImg1TNFromLast[i].cols;
            movObjDistTNtoReal[i] = vector<double>((size_t)movObjCorrsImg1TNFromLast[i].cols, fakeDistTNCorrespondences);
        } else if (!movObjCorrsImg2TNFromLast[i].empty()) {
            missingCImg1[i] = movObjCorrsImg2TNFromLast[i].cols;
            movObjDistTNtoReal[i] = vector<double>((size_t)movObjCorrsImg2TNFromLast[i].cols, fakeDistTNCorrespondences);
        }
    }

    //Get ROIs of moving objects by calculating the simplified contour (non-convex) of every object
    vector<Rect> objROIs(actNrMovObj);
    for (size_t i = 0; i < actNrMovObj; i++) {
        vector<Point> hull;
        genHullFromMask(movObjLabelsFromLast[i], hull);
        objROIs[i] = boundingRect(hull);
    }

    //Get missing TN correspondences for found keypoints
    std::uniform_int_distribution<int32_t> distributionX2(0, imgSize.width - 1);
    std::uniform_int_distribution<int32_t> distributionY2(0, imgSize.height - 1);
    for (size_t i = 0; i < actNrMovObj; i++) {
        if (missingCImg2[i] > 0) {
            //Enlarge mask
            Mat movObjMaskFromLast2Border(movObjMaskFromLastLarge2[i].size(), movObjMaskFromLastLarge2[i].type());
            cv::copyMakeBorder(movObjMaskFromLast2, movObjMaskFromLast2Border, posadd, posadd, posadd, posadd,
                               BORDER_CONSTANT, cv::Scalar(0));
            Mat elemnew = Mat::ones(missingCImg2[i], 3, CV_64FC1);
            int cnt1 = 0;
            for (int j = 0; j < missingCImg2[i]; j++) {
                int cnt = 0;
                while (cnt < maxCnt) {
                    Point_<int32_t> pt = Point_<int32_t>(distributionX2(rand_gen), distributionY2(rand_gen));
                    Mat s_tmp = movObjMaskFromLast2Border(Rect(pt, Size(sqrSi, sqrSi)));
                    if (s_tmp.at<unsigned char>(posadd, posadd) == 0) {
//                        csurr.copyTo(s_tmp);
                        s_tmp.at<unsigned char>(posadd, posadd) = 1;
                        elemnew.at<double>(j, 0) = (double) pt.x;
                        elemnew.at<double>(j, 1) = (double) pt.y;
                        break;
                    }
                    cnt++;
                }
                if (cnt == maxCnt)
                    break;
                cnt1++;
            }
            if (cnt1 > 0) {
                Mat dispMask;
                if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                    dispMask = (movObjMaskFromLast2 > 0);
                }

                movObjMaskFromLast2 |= movObjMaskFromLast2Border(Rect(Point(posadd, posadd), imgSize));

                if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                    Mat dispMask2 = (movObjMaskFromLast2 > 0);
                    vector<Mat> channels;
                    Mat b = Mat::zeros(dispMask2.size(), CV_8UC1);
                    channels.push_back(b);
                    channels.push_back(dispMask);
                    channels.push_back(dispMask2);
                    Mat img3c;
                    merge(channels, img3c);
                    if(!writeIntermediateImg(img3c, "random_TN_in_img2_for_backprojected_moving_obj_TN_of_img1")){
                        namedWindow("Random TN in img2 for backprojected moving object TN of img1", WINDOW_AUTOSIZE);
                        imshow("Random TN in img2 for backprojected moving object TN of img1", img3c);
                        waitKey(0);
                        destroyWindow("Random TN in img2 for backprojected moving object TN of img1");
                    }
                }

                if (movObjCorrsImg2TNFromLast[i].empty()){
                    elemnew.rowRange(0, cnt1).copyTo(movObjCorrsImg2TNFromLast[i]);
                }else {
                    movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].t();
                    movObjCorrsImg2TNFromLast[i].push_back(elemnew.rowRange(0, cnt1));
                }
                movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].t();
                missingCImg2[i] -= cnt1;
//                movObjDistTNtoReal[i] = vector<double>(cnt1, fakeDistTNCorrespondences);
            }
            if (missingCImg2[i] > 0) {
                movObjDistTNtoReal[i].erase(movObjDistTNtoReal[i].begin() + movObjCorrsImg1TNFromLast[i].cols -
                                            missingCImg2[i], movObjDistTNtoReal[i].end());
                movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].colRange(0,
                                                                                     movObjCorrsImg1TNFromLast[i].cols -
                                                                                     missingCImg2[i]);
            }
        } else if (missingCImg1[i] > 0) {
            Mat elemnew = Mat::ones(missingCImg1[i], 3, CV_64FC1);
            int cnt1 = 0;
            std::uniform_int_distribution<int32_t> distributionX(objROIs[i].x, objROIs[i].x + objROIs[i].width - 1);
            std::uniform_int_distribution<int32_t> distributionY(objROIs[i].y,
                                                                 objROIs[i].y + objROIs[i].height - 1);
            //Enlarge mask
            Mat movObjMaskFromLastBorder(movObjMaskFromLastLarge[i].size(), movObjMaskFromLastLarge[i].type());
            cv::copyMakeBorder(movObjMaskFromLastOld, movObjMaskFromLastBorder, posadd, posadd, posadd, posadd,
                               BORDER_CONSTANT, cv::Scalar(0));
            for (int j = 0; j < missingCImg1[i]; j++) {
                int cnt = 0;
                while (cnt < maxCnt) {
                    Point_<int32_t> pt = Point_<int32_t>(distributionX(rand_gen), distributionY(rand_gen));
                    Mat s_tmp = movObjMaskFromLastBorder(Rect(pt, Size(sqrSi, sqrSi)));
                    if ((s_tmp.at<unsigned char>(posadd, posadd) == 0) &&
                        (movObjLabelsFromLast[i].at<unsigned char>(pt) > 0)) {
                        csurr.copyTo(s_tmp);
                        Mat s_tmpAdd = movObjMaskFromLastLargeAdd[i](Rect(pt, Size(sqrSi, sqrSi)));
                        s_tmpAdd += csurr;
                        elemnew.at<double>(j, 0) = (double) pt.x;
                        elemnew.at<double>(j, 1) = (double) pt.y;
                        break;
                    }
                    cnt++;
                }
                if (cnt == maxCnt)
                    break;
                cnt1++;
            }
            if (cnt1 > 0) {
                Mat dispMask;
                if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                    dispMask = (movObjMaskFromLastOld > 0);
                }

                movObjMaskFromLastOld |= movObjMaskFromLastBorder(Rect(Point(posadd, posadd), imgSize));

                if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                    Mat dispMask2 = (movObjMaskFromLastOld > 0);
                    vector<Mat> channels;
                    Mat b = Mat::zeros(dispMask2.size(), CV_8UC1);
                    channels.push_back(b);
                    channels.push_back(dispMask);
                    channels.push_back(dispMask2);
                    Mat img3c;
                    merge(channels, img3c);
                    if(!writeIntermediateImg(img3c, "random_TN_in_img1_for_backprojected_moving_obj_TN_of_img2")){
                        namedWindow("Random TN in img1 for backprojected moving object TN of img2", WINDOW_AUTOSIZE);
                        imshow("Random TN in img1 for backprojected moving object TN of img2", img3c);
                        waitKey(0);
                        destroyWindow("Random TN in img1 for backprojected moving object TN of img2");
                    }
                }

//                movObjLabelsFromLast[i] |= movObjMaskFromLastBorder(Rect(Point(posadd, posadd), imgSize));
                if(movObjCorrsImg1TNFromLast[i].empty()){
                    elemnew.rowRange(0, cnt1).copyTo(movObjCorrsImg1TNFromLast[i]);
                }else {
                    movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].t();
                    movObjCorrsImg1TNFromLast[i].push_back(elemnew.rowRange(0, cnt1));
                }
                movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].t();
                missingCImg1[i] -= cnt1;
//                movObjDistTNtoReal[i] = vector<double>(cnt1, fakeDistTNCorrespondences);
            }
            if (missingCImg1[i] > 0) {
                movObjDistTNtoReal[i].erase(movObjDistTNtoReal[i].begin() + movObjCorrsImg2TNFromLast[i].cols -
                                            missingCImg1[i], movObjDistTNtoReal[i].end());
                movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].colRange(0,
                                                                                     movObjCorrsImg2TNFromLast[i].cols -
                                                                                     missingCImg1[i]);
            }
        }
    }

    //Additionally add TN using the inlier ratio
    for (size_t i = 0; i < actNrMovObj; i++) {
        if (missingCImg2[i] == 0)//Skip adding TN if adding TN in the second image failed already before
        {
            /*missingCImg2[i] =
                    (int) round((double) movObjCorrsImg1TPFromLast[i].cols * (1.0 - inlRat[actFrameCnt])) -
                    movObjCorrsImg1TNFromLast[i].cols;*/
            missingCImg2[i] =
                    (int) round((double) movObjCorrsImg1TPFromLast[i].cols * (1.0 / inlRat[actFrameCnt] - 1.0)) -
                    movObjCorrsImg1TNFromLast[i].cols;
            if(missingCImg2[i] < 0){
                cout << "There are still too many TN (+" << -1*missingCImg2[i] << ") on the backprojected moving object #" << i << " in image 1!" << endl;
            }
        } else {
            missingCImg2[i] = -1;
        }
    }
    element = cv::getStructuringElement(MORPH_RECT, Size(sqrSi, sqrSi));
    //Enlarge mask for image 2
    Mat movObjMaskFromLast2Border(movObjMaskFromLastLarge2[0].size(), movObjMaskFromLastLarge2[0].type());
    cv::copyMakeBorder(movObjMaskFromLast2, movObjMaskFromLast2Border, posadd, posadd, posadd, posadd,
                       BORDER_CONSTANT,
                       cv::Scalar(0));
    //Enlarge mask for image 1
    Mat movObjMaskFromLastBorder(movObjMaskFromLastLarge[0].size(), movObjMaskFromLastLarge[0].type());
    cv::copyMakeBorder(movObjMaskFromLastOld, movObjMaskFromLastBorder, posadd, posadd, posadd, posadd,
                       BORDER_CONSTANT,
                       cv::Scalar(0));
    for (size_t i = 0; i < actNrMovObj; i++) {
        //Generate a depth map for generating TN based on the depth of the back-projected 3D points
        double minDepth = DBL_MAX, maxDepth = DBL_MIN;
        for (size_t j = 0; j < movObjCorrsImg12TPFromLast_Idx[i].size(); j++) {
            double sDepth = movObj3DPtsCam[i][movObjCorrsImg12TPFromLast_Idx[i][j]].z;
            if (sDepth < minDepth)
                minDepth = sDepth;
            if (sDepth > maxDepth)
                maxDepth = sDepth;
        }
        Mat randDepth(imgSize, CV_64FC1);
        randu(randDepth, Scalar(minDepth), Scalar(maxDepth + 0.001));

        std::uniform_int_distribution<int32_t> distributionX(objROIs[i].x, objROIs[i].x + objROIs[i].width - 1);
        std::uniform_int_distribution<int32_t> distributionY(objROIs[i].y, objROIs[i].y + objROIs[i].height - 1);
        int areaMO = cv::countNonZero(movObjLabelsFromLast[i]);
        int cnt2 = 0;
        while (((areaMO < maxOArea) || (cnt2 == 0)) && (missingCImg2[i] > 0) &&
               (cnt2 < maxCnt))//If not all elements could be selected, try to enlarge the area
        {
            //Generate label mask for image 1
            Mat movObjLabelsFromLastN;
            cv::copyMakeBorder(movObjLabelsFromLast[i], movObjLabelsFromLastN, posadd, posadd, posadd, posadd,
                               BORDER_CONSTANT, cv::Scalar(0));
            movObjLabelsFromLastN = (movObjLabelsFromLastN == 0);
            movObjLabelsFromLastN |= movObjMaskFromLastBorder;

            Mat dispMask1, dispMask2;
            if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                dispMask1 = (movObjLabelsFromLastN > 0);
                dispMask2 = (movObjMaskFromLast2Border > 0);
            }

            std::vector<cv::Point2d> x1TN;
            std::vector<cv::Point2d> x2TN;
            int32_t remainingTN = genTrueNegCorrs(missingCImg2[i],
                                                  distributionX,
                                                  distributionY,
                                                  distributionX2,
                                                  distributionY2,
                                                  x1TN,
                                                  x2TN,
                                                  movObjDistTNtoReal[i],
                                                  movObjLabelsFromLastN,
                                                  movObjMaskFromLast2Border,
                                                  randDepth);

            cnt2++;
            if (remainingTN != missingCImg2[i]) {

                if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                    Mat dispMask12 = (movObjLabelsFromLastN > 0);
                    Mat dispMask22 = (movObjMaskFromLast2Border > 0);
                    vector<Mat> channels;
                    Mat b = Mat::zeros(dispMask12.size(), CV_8UC1);
                    channels.push_back(b);
                    channels.push_back(dispMask1);
                    channels.push_back(dispMask12);
                    Mat img3c;
                    merge(channels, img3c);
                    bool wii = !writeIntermediateImg(img3c, "random_TN_in_img1_for_backprojected_moving_obj");
                    if(wii){
                        namedWindow("Random TN in img1 for backprojected moving object", WINDOW_AUTOSIZE);
                        imshow("Random TN in img1 for backprojected moving object", img3c);
                    }
                    channels.clear();
                    channels.push_back(b);
                    channels.push_back(dispMask2);
                    channels.push_back(dispMask22);
                    merge(channels, img3c);
                    wii |= !writeIntermediateImg(img3c, "random_TN_in_img2_for_backprojected_moving_obj");
                    if(wii){
                        namedWindow("Random TN in img2 for backprojected moving object", WINDOW_AUTOSIZE);
                        imshow("Random TN in img2 for backprojected moving object", img3c);
                        waitKey(0);
                        destroyWindow("Random TN in img1 for backprojected moving object");
                        destroyWindow("Random TN in img2 for backprojected moving object");
                    }
                }

                movObjLabelsFromLastN(Rect(Point(posadd, posadd), imgSize)) &= movObjLabelsFromLast[i];
                movObjMaskFromLastBorder(Rect(Point(posadd, posadd), imgSize)) |= movObjLabelsFromLastN(
                        Rect(Point(posadd, posadd), imgSize));

                auto nelem = (int32_t) x1TN.size();
                Mat elemnew = Mat::ones(nelem, 3, CV_64FC1);
                Mat elemnew2 = Mat::ones(nelem, 3, CV_64FC1);
                for (int32_t j = 0; j < nelem; j++) {
                    elemnew.at<double>(j, 0) = x1TN[j].x;
                    elemnew.at<double>(j, 1) = x1TN[j].y;
                    elemnew2.at<double>(j, 0) = x2TN[j].x;
                    elemnew2.at<double>(j, 1) = x2TN[j].y;
                }
                if (movObjCorrsImg1TNFromLast[i].empty()){
                    elemnew.copyTo(movObjCorrsImg1TNFromLast[i]);
                }else {
                    movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].t();
                    movObjCorrsImg1TNFromLast[i].push_back(elemnew);
                }
                movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].t();
                if (movObjCorrsImg2TNFromLast[i].empty()){
                    elemnew2.copyTo(movObjCorrsImg2TNFromLast[i]);
                }else {
                    movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].t();
                    movObjCorrsImg2TNFromLast[i].push_back(elemnew2);
                }
                movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].t();
            }
            if (remainingTN > 0) {
                //Perform dilation
                Mat imgSDdilate;
                dilate(movObjLabelsFromLast[i], imgSDdilate, element);
                imgSDdilate &= ((movObjMaskFromLast == 0) | movObjLabelsFromLast[i]);
                int areaMO2 = cv::countNonZero(imgSDdilate);
                if (areaMO2 > areaMO) {
                    areaMO = areaMO2;
                    if (areaMO < maxOArea) {
                        imgSDdilate.copyTo(movObjLabelsFromLast[i]);
                        objROIs[i] = Rect(max(objROIs[i].x - posadd, 0), max(objROIs[i].y - posadd, 0),
                                          objROIs[i].width + 2 * posadd, objROIs[i].height + 2 * posadd);
                        objROIs[i] = Rect(objROIs[i].x, objROIs[i].y,
                                          (objROIs[i].x + objROIs[i].width) > imgSize.width ? (imgSize.width -
                                                                                               objROIs[i].x)
                                                                                            : objROIs[i].width,
                                          (objROIs[i].y + objROIs[i].height) > imgSize.height ? (imgSize.height -
                                                                                                 objROIs[i].y)
                                                                                              : objROIs[i].height);
                        distributionX = std::uniform_int_distribution<int32_t>(objROIs[i].x,
                                                                               objROIs[i].x + objROIs[i].width - 1);
                        distributionY = std::uniform_int_distribution<int32_t>(objROIs[i].y,
                                                                               objROIs[i].y + objROIs[i].height - 1);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            missingCImg2[i] = remainingTN;
        }
    }
    movObjMaskFromLastBorder(Rect(Point(posadd, posadd), imgSize)).copyTo(movObjMaskFromLastOld);
    movObjMaskFromLast2 |= movObjMaskFromLast2Border(Rect(Point(posadd, posadd), imgSize));

    movObjMaskFromLast = Mat::zeros(imgSize, CV_8UC1);
    actTNPerMovObjFromLast.resize(actNrMovObj);
    for (size_t i = 0; i < actNrMovObj; i++) {
        //Generate a final non-convex hull or contour for every moving object
        genHullFromMask(movObjLabelsFromLast[i], convhullPtsObj[i]);
        movObjMaskFromLast |= (unsigned char) (i + 1) * movObjLabelsFromLast[i];
        //actAreaMovObj[i] = contourArea(convhullPtsObj[i]);

        actTNPerMovObjFromLast[i] = movObjCorrsImg1TNFromLast[i].cols;

        actTruePosOnMovObjFromLast += movObjCorrsImg1TPFromLast[i].cols;
        actTrueNegOnMovObjFromLast += actTNPerMovObjFromLast[i];
    }
    actCorrsOnMovObjFromLast = actTrueNegOnMovObjFromLast + actTruePosOnMovObjFromLast;

    movObj3DPtsWorldAllFrames.insert(movObj3DPtsWorldAllFrames.end(), movObj3DPtsWorld.begin(), movObj3DPtsWorld.end());

    //Finally visualize the labels
    if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
        //Generate colormap for moving obejcts (every object has a different color)
        Mat colors = Mat((int)actNrMovObj, 1, CV_8UC1);
        unsigned char addc = actNrMovObj > 255 ? (unsigned char)255 : (unsigned char) actNrMovObj;
        addc = addc < (unsigned char)2 ? (unsigned char)255 : ((unsigned char)255 / (addc - (unsigned char)1));
        colors.at<unsigned char>(0) = 0;
        for (int k = 1; k < (int)actNrMovObj; ++k) {
            colors.at<unsigned char>(k) = colors.at<unsigned char>(k - 1) + addc;
        }
        Mat colormap_img;
        applyColorMap(colors, colormap_img, COLORMAP_PARULA);
        Mat labelImgRGB = Mat::zeros(imgSize, CV_8UC3);
        for (size_t i = 0; i < actNrMovObj; i++) {
            for (int r = 0; r < imgSize.height; r++) {
                for (int c = 0; c < imgSize.width; c++) {
                    if (movObjLabelsFromLast[i].at<unsigned char>(r, c) != 0) {
                        labelImgRGB.at<cv::Vec3b>(r, c) = colormap_img.at<cv::Vec3b>(i);
                    }
                }
            }
        }
        if(!writeIntermediateImg(labelImgRGB, "final_backprojected_moving_obj_labels")){
            namedWindow("Backprojected final moving object labels", WINDOW_AUTOSIZE);
            imshow("Backprojected final moving object labels", labelImgRGB);
            waitKey(0);
            destroyWindow("Backprojected final moving object labels");
        }
    }
}

void genStereoSequ::genHullFromMask(const cv::Mat &mask, std::vector<cv::Point> &finalHull) {

    if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
        if(!writeIntermediateImg(mask > 0, "original_backprojected_moving_obj_mask")){
            namedWindow("Original backprojected moving object mask", WINDOW_AUTOSIZE);
            imshow("Original backprojected moving object mask", mask > 0);
            waitKey(0);
        }
    }

    //Get the contour of the mask
    vector<vector<Point>> contours;
    Mat finalMcopy = mask.clone();
    findContours(finalMcopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    //Calculate 1 single outer contour for multiple sub-objects avoiding a convex hull
    int contSize = (int) contours.size();
    if (contSize > 1) {
        //Get biggest element (which will be the main element)
        double biggestArea = 0;
        int idx = 0;
        for (int i = 0; i < contSize; ++i) {
            double actArea = contourArea(contours[i]);
            if (actArea > biggestArea) {
                biggestArea = actArea;
                idx = i;
            }
        }
        //Calculate the center of mass of every element to get the distances between elements
        Point2f bigAreaCenter;
        vector<Point> bigAreaContour = contours[idx];
        cv::Moments bigAreaMoments = moments(bigAreaContour, true);
        bigAreaCenter = Point2f((float)(bigAreaMoments.m10 / bigAreaMoments.m00),
                                (float)(bigAreaMoments.m01 / bigAreaMoments.m00));
        vector<pair<int, Point2f>> areaCenters;
        for (int i = 0; i < contSize; ++i) {
            if (i == idx) continue;
            cv::Moments areaMoment = moments(contours[i], true);
            areaCenters.emplace_back(
                    make_pair(i, Point2f((float)(areaMoment.m10 / areaMoment.m00),
                                         (float)(areaMoment.m01 / areaMoment.m00))));
        }
        //Begin with the nearest element and combine every 2 nearest elements
        for (int i = 0; i < contSize - 1; ++i) {
            float minDist = FLT_MAX;
            pair<int, int> minDistIdx;
            for (int j = 0; j < (int)areaCenters.size(); ++j) {
                Point2f ptdiff = bigAreaCenter - areaCenters[j].second;
                float dist = sqrt(ptdiff.x * ptdiff.x + ptdiff.y * ptdiff.y);
                if (dist < minDist) {
                    minDist = dist;
                    minDistIdx = make_pair(areaCenters[j].first, j);
                }
            }
            //Combine nearest element and biggest element
            int maxBOSi = (int) bigAreaContour.size();
            vector<int> hullIdxs;
            vector<Point> hullPts1, hullPts2;
            vector<Point> comb2Areas = bigAreaContour;
            comb2Areas.insert(comb2Areas.end(), contours[minDistIdx.first].begin(), contours[minDistIdx.first].end());
            convexHull(comb2Areas, hullIdxs);
            //Check from which area the convex hull points are
            int hullIdxInsert[2][2] = {{-1, -1},
                                       {-1, -1}};
            bool advIdx[2][2] = {{false, false},
                                 {false, false}};
            bool branchChk[4] = {false, false, false, false};
            for (int k = 0; k < (int)hullIdxs.size(); ++k) {
                if (hullIdxs[k] < maxBOSi) {
                    branchChk[0] = true;
                    if (branchChk[0] && branchChk[1]) {
                        branchChk[3] = true;
                        if (hullIdxInsert[0][0] < 0) {
                            hullIdxInsert[0][0] = hullIdxs[k];
                        }
                        if (hullIdxInsert[1][0] < 0) {
                            hullIdxInsert[1][0] = hullIdxs[k - 1] - maxBOSi;
                            advIdx[1][0] = true;
                        }
                    }
                    if (branchChk[2]) {
                        if (hullIdxInsert[0][1] < 0) {
                            hullIdxInsert[0][1] = hullIdxs[k];
                        }
                        if ((hullIdxInsert[1][1] < 0) && (!hullPts2.empty())) {
                            hullIdxInsert[1][1] = hullIdxs[k - 1] - maxBOSi;
                            advIdx[1][1] = true;
                        }
                    }

                    hullPts1.push_back(comb2Areas[hullIdxs[k]]);
                } else {
                    branchChk[1] = true;
                    if (branchChk[0] && branchChk[1]) {
                        branchChk[2] = true;
                        if (hullIdxInsert[0][0] < 0) {
                            hullIdxInsert[0][0] = hullIdxs[k - 1];
                            advIdx[0][0] = true;
                        }
                        if (hullIdxInsert[1][0] < 0) {
                            hullIdxInsert[1][0] = hullIdxs[k] - maxBOSi;
                        }
                    }
                    if (branchChk[3]) {
                        if (hullIdxInsert[0][1] < 0) {
                            hullIdxInsert[0][1] = hullIdxs[k - 1];
                            advIdx[0][1] = true;
                        }
                        if (hullIdxInsert[1][1] < 0) {
                            hullIdxInsert[1][1] = hullIdxs[k] - maxBOSi;
                        }
                    }
                    hullPts2.push_back(comb2Areas[hullIdxs[k]]);
                }
            }

            if (!hullPts2.empty()) {

                /*if(verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
                    Mat maskcontours = Mat::zeros(imgSize, CV_8UC3);
                    vector<vector<Point>> tmp(1);
                    tmp[0] = bigAreaContour;
                    drawContours(maskcontours, tmp, 0, Scalar(0, 255, 0));
                    namedWindow("New big area", WINDOW_AUTOSIZE);
                    imshow("New big area", maskcontours);

                    waitKey(0);
                    destroyWindow("New big area");
                }*/

                if (hullIdxInsert[0][1] < 0) {
                    if (advIdx[0][0]) {
                        hullIdxInsert[0][1] = hullIdxs[0];
                        hullIdxInsert[1][1] = hullIdxs.back() - maxBOSi;
                        advIdx[1][1] = true;
                    } else {
                        hullIdxInsert[1][1] = hullIdxs[0] - maxBOSi;
                        hullIdxInsert[0][1] = hullIdxs.back();
                        advIdx[0][1] = true;
                    }
                }

                CV_Assert((advIdx[0][0] ^ advIdx[1][0]) && (advIdx[0][1] ^ advIdx[1][1]) &&
                          (advIdx[0][0] ^ advIdx[0][1]));

                //Extract for each area both possible contour elements
                vector<Point> bigAreaContourNew1, bigAreaContourNew2, bigAreaContourNew12, bigAreaContourNew22;
                if (advIdx[0][0]) {
                    if (hullIdxInsert[0][1] > hullIdxInsert[0][0]) {
                        getFirstPartContourNeg(bigAreaContourNew1, bigAreaContour, hullIdxInsert[0][1],
                                               hullIdxInsert[0][0]);
                        getSecPartContourNeg(bigAreaContourNew12, bigAreaContour, hullIdxInsert[0][1],
                                             hullIdxInsert[0][0]);
                    } else {
                        getFirstPartContourPos(bigAreaContourNew1, bigAreaContour, hullIdxInsert[0][1],
                                               hullIdxInsert[0][0]);
                        getSecPartContourPos(bigAreaContourNew12, bigAreaContour, hullIdxInsert[0][1],
                                             hullIdxInsert[0][0]);
                    }

                    if (hullIdxInsert[1][0] > hullIdxInsert[1][1]) {
                        getFirstPartContourNeg(bigAreaContourNew2, contours[minDistIdx.first], hullIdxInsert[1][0],
                                               hullIdxInsert[1][1]);
                        getSecPartContourNeg(bigAreaContourNew22, contours[minDistIdx.first], hullIdxInsert[1][0],
                                             hullIdxInsert[1][1]);
                    } else {
                        getFirstPartContourPos(bigAreaContourNew2, contours[minDistIdx.first], hullIdxInsert[1][0],
                                               hullIdxInsert[1][1]);
                        getSecPartContourPos(bigAreaContourNew22, contours[minDistIdx.first], hullIdxInsert[1][0],
                                             hullIdxInsert[1][1]);
                    }
                } else {
                    if (hullIdxInsert[0][0] > hullIdxInsert[0][1]) {
                        getFirstPartContourNeg(bigAreaContourNew1, bigAreaContour, hullIdxInsert[0][0],
                                               hullIdxInsert[0][1]);
                        getSecPartContourNeg(bigAreaContourNew12, bigAreaContour, hullIdxInsert[0][0],
                                             hullIdxInsert[0][1]);
                    } else {
                        getFirstPartContourPos(bigAreaContourNew1, bigAreaContour, hullIdxInsert[0][0],
                                               hullIdxInsert[0][1]);
                        getSecPartContourPos(bigAreaContourNew12, bigAreaContour, hullIdxInsert[0][0],
                                             hullIdxInsert[0][1]);
                    }

                    if (hullIdxInsert[1][1] > hullIdxInsert[1][0]) {
                        getFirstPartContourNeg(bigAreaContourNew2, contours[minDistIdx.first], hullIdxInsert[1][1],
                                               hullIdxInsert[1][0]);
                        getSecPartContourNeg(bigAreaContourNew22, contours[minDistIdx.first], hullIdxInsert[1][1],
                                             hullIdxInsert[1][0]);
                    } else {
                        getFirstPartContourPos(bigAreaContourNew2, contours[minDistIdx.first], hullIdxInsert[1][1],
                                               hullIdxInsert[1][0]);
                        getSecPartContourPos(bigAreaContourNew22, contours[minDistIdx.first], hullIdxInsert[1][1],
                                             hullIdxInsert[1][0]);
                    }
                }

                //Select the correct contours of both seperate areas which offer together the largest area and/or have points of the convex hull in common
                if ((hullPts1.size() <= 2) || (hullPts2.size() <= 2)) {
                    double areas[4] = {0, 0, 0, 0};
                    vector<Point> testCont[4];
                    testCont[0] = bigAreaContourNew1;
                    testCont[0].insert(testCont[0].end(), bigAreaContourNew2.begin(), bigAreaContourNew2.end());
                    areas[0] = cv::contourArea(testCont[0]);
                    testCont[1] = bigAreaContourNew1;
                    testCont[1].insert(testCont[1].end(), bigAreaContourNew22.begin(), bigAreaContourNew22.end());
                    areas[1] = cv::contourArea(testCont[1]);
                    testCont[2] = bigAreaContourNew12;
                    testCont[2].insert(testCont[2].end(), bigAreaContourNew2.begin(), bigAreaContourNew2.end());
                    areas[2] = cv::contourArea(testCont[2]);
                    testCont[3] = bigAreaContourNew12;
                    testCont[3].insert(testCont[3].end(), bigAreaContourNew22.begin(), bigAreaContourNew22.end());
                    areas[3] = cv::contourArea(testCont[3]);
                    std::ptrdiff_t pdiff = max_element(areas, areas + 4) - areas;

                    bool selfintersection = true;
                    bool hullUsedasOutPt = false;
                    while (selfintersection) {
                        Rect conRe;
                        conRe = cv::boundingRect(testCont[pdiff]);
                        Point conReSP = Point(conRe.x, conRe.y);
                        Mat testMat = Mat::zeros(conRe.height, conRe.width, CV_8UC1);
                        for (int j = 1; j < (int) testCont[pdiff].size(); ++j) {
                            Mat testMat1 = Mat::zeros(conRe.height, conRe.width, CV_8UC1);
                            cv::line(testMat1, testCont[pdiff][j - 1] - conReSP, testCont[pdiff][j] - conReSP,
                                     Scalar(1));
                            testMat1.at<unsigned char>(testCont[pdiff][j] - conReSP) = 0;
                            testMat += testMat1;
                        }
                        Mat testMat1 = Mat::zeros(conRe.height, conRe.width, CV_8UC1);
                        cv::line(testMat1, testCont[pdiff].back() - conReSP, testCont[pdiff][0] - conReSP, Scalar(1));
                        testMat1.at<unsigned char>(testCont[pdiff][0] - conReSP) = 0;
                        testMat += testMat1;

                        /*namedWindow("Line intersections", WINDOW_AUTOSIZE);
                        imshow("Line intersections", testMat > 0);
                        namedWindow("Line intersections1", WINDOW_AUTOSIZE);
                        imshow("Line intersections1", testMat > 1);

                        waitKey(0);
                        destroyWindow("Line intersections");
                        destroyWindow("Line intersections1");*/

                        bool foundIntSec = false;
                        for (int k = 0; k < conRe.height; ++k) {
                            for (int l = 0; l < conRe.width; ++l) {
                                if (testMat.at<unsigned char>(k, l) > 1) {
                                    foundIntSec = true;
                                    break;
                                }
                            }
                            if (foundIntSec) break;
                        }
                        if (foundIntSec) {
                            areas[pdiff] = 0;
                            pdiff = max_element(areas, areas + 4) - areas;
                            if(areas[pdiff] == 0){
                                if((hullPts1.size() <= 2) && (hullPts2.size() > 2) && !hullUsedasOutPt){
                                    testCont[1] = hullPts1;
                                    testCont[1].insert(testCont[1].end(), bigAreaContourNew22.begin(), bigAreaContourNew22.end());
                                    areas[1] = cv::contourArea(testCont[1]);
                                    testCont[3] = hullPts1;
                                    testCont[3].insert(testCont[3].end(), bigAreaContourNew22.begin(), bigAreaContourNew22.end());
                                    areas[3] = cv::contourArea(testCont[3]);
                                    pdiff = max_element(areas, areas + 4) - areas;
                                }
                                else if((hullPts1.size() > 2) && (hullPts2.size() <= 2) && !hullUsedasOutPt){
                                    testCont[0] = bigAreaContourNew1;
                                    testCont[0].insert(testCont[0].end(), hullPts2.begin(), hullPts2.end());
                                    areas[0] = cv::contourArea(testCont[0]);
                                    testCont[2] = bigAreaContourNew12;
                                    testCont[2].insert(testCont[2].end(), hullPts2.begin(), hullPts2.end());
                                    areas[2] = cv::contourArea(testCont[2]);
                                    pdiff = max_element(areas, areas + 4) - areas;
                                }
                                else{
                                    testCont[0].clear();
                                    testCont[0].resize(hullIdxs.size());
                                    for (size_t j = 0; j < hullIdxs.size(); ++j) {
                                        testCont[0][j] = comb2Areas[hullIdxs[j]];
                                    }
                                    pdiff = 0;
                                    selfintersection = false;
                                }
                                hullUsedasOutPt = true;
                            }
                        } else {
                            selfintersection = false;
                        }
                    }
                    bigAreaContour = testCont[pdiff];
                } else {
                    bool selCont[4] = {false, false, false, false};
                    int equCnt[2] = {0, 0};
                    for (auto& j : hullPts1) {
                        for (auto& k : bigAreaContourNew1) {
                            if ((abs(j.x - k.x) +
                                 abs(j.y - k.y)) == 0) {
                                equCnt[0]++;
                                if (equCnt[0] > 2) {
                                    selCont[0] = true;
                                    break;
                                }
                            }
                        }
                        if (selCont[0]) break;
                        for (auto& k : bigAreaContourNew12) {
                            if ((abs(j.x - k.x) +
                                 abs(j.y - k.y)) == 0) {
                                equCnt[1]++;
                                if (equCnt[1] > 2) {
                                    selCont[1] = true;
                                    break;
                                }
                            }
                        }
                        if (selCont[1]) break;
                    }
                    equCnt[0] = 0;
                    equCnt[1] = 0;
                    for (auto& j : hullPts2) {
                        for (auto& k : bigAreaContourNew2) {
                            if ((abs(j.x - k.x) +
                                 abs(j.y - k.y)) == 0) {
                                equCnt[0]++;
                                if (equCnt[0] > 2) {
                                    selCont[2] = true;
                                    break;
                                }
                            }
                        }
                        if (selCont[2]) break;
                        for (auto& k : bigAreaContourNew22) {
                            if ((abs(j.x - k.x) +
                                 abs(j.y - k.y)) == 0) {
                                equCnt[1]++;
                                if (equCnt[1] > 2) {
                                    selCont[3] = true;
                                    break;
                                }
                            }
                        }
                        if (selCont[3]) break;
                    }
                    if (selCont[0] && selCont[2]) {
                        bigAreaContour = bigAreaContourNew1;
                        bigAreaContour.insert(bigAreaContour.end(), bigAreaContourNew2.begin(),
                                              bigAreaContourNew2.end());
                    } else if (selCont[0] && selCont[3]) {
                        bigAreaContour = bigAreaContourNew1;
                        bigAreaContour.insert(bigAreaContour.end(), bigAreaContourNew22.begin(),
                                              bigAreaContourNew22.end());
                    } else if (selCont[1] && selCont[2]) {
                        bigAreaContour = bigAreaContourNew12;
                        bigAreaContour.insert(bigAreaContour.end(), bigAreaContourNew2.begin(),
                                              bigAreaContourNew2.end());
                    } else {
                        bigAreaContour = bigAreaContourNew12;
                        bigAreaContour.insert(bigAreaContour.end(), bigAreaContourNew22.begin(),
                                              bigAreaContourNew22.end());
                    }
                }

                //Calculate the new center of the big area
                bigAreaMoments = moments(bigAreaContour, true);
                bigAreaCenter = Point2f((float)(bigAreaMoments.m10 / bigAreaMoments.m00),
                                        (float)(bigAreaMoments.m01 / bigAreaMoments.m00));
            }

            //Delete the used area center
            areaCenters.erase(areaCenters.begin() + minDistIdx.second);
        }
        contours.clear();
        contours.resize(1);
        contours[0] = bigAreaContour;
    }else if(contSize == 0){
        if(verbose & SHOW_IMGS_AT_ERROR) {
            if(!writeIntermediateImg(mask > 0, "error_no_contour_found")){
                namedWindow("Error - No contour found", WINDOW_AUTOSIZE);
                imshow("Error - No contour found", mask > 0);
                waitKey(0);
                destroyWindow("Error - No contour found");
            }
        }
        throw SequenceException("No contour found for backprojected moving object!");
    }

    //Simplify the contour
    double epsilon = 0.005 * cv::arcLength(contours[0], true);//0.5% of the overall contour length
    if (epsilon > 6.5) {
        epsilon = 6.5;
    } else if (epsilon < 2.0) {
        epsilon = 2.0;
    }
    approxPolyDP(contours[0], finalHull, epsilon, true);

    if (verbose & SHOW_BACKPROJECT_MOV_OBJ_CORRS) {
        Mat maskcontours = Mat::zeros(imgSize, CV_8UC3);
        drawContours(maskcontours, contours, 0, Scalar(0, 255, 0));
        vector<vector<Point>> tmp(1);
        tmp[0] = finalHull;
        drawContours(maskcontours, tmp, 0, Scalar(0, 0, 255));
        if(!writeIntermediateImg(maskcontours > 0, "approximated_and_original_backprojected_moving_object_mask_contour")){
            namedWindow("Approximated and original backprojected moving object mask contour", WINDOW_AUTOSIZE);
            imshow("Approximated and original backprojected moving object mask contour", maskcontours > 0);

            waitKey(0);
            destroyWindow("Original backprojected moving object mask");
            destroyWindow("Approximated and original backprojected moving object mask contour");
        }
    }
}

void genStereoSequ::genMovObjHulls(const cv::Mat &corrMask, std::vector<cv::Point> &kps, cv::Mat &finalMask,
                                   std::vector<cv::Point> *hullPts) {
    int sqrSi = csurr.rows;

    //Get the convex hull of the keypoints
    vector<vector<Point>> hull(1);
    convexHull(kps, hull[0]);

    if (hullPts) {
        *hullPts = hull[0];
    }

    //Get bounding box
    Rect hullBB = boundingRect(hull[0]);
    hullBB = Rect(max(hullBB.x - sqrSi, 0), max(hullBB.y - sqrSi, 0), hullBB.width + 2 * sqrSi,
                  hullBB.height + 2 * sqrSi);
    hullBB = Rect(hullBB.x, hullBB.y,
                  (hullBB.x + hullBB.width) > imgSize.width ? (imgSize.width - hullBB.x) : hullBB.width,
                  (hullBB.y + hullBB.height) > imgSize.height ? (imgSize.height - hullBB.y) : hullBB.height);

    //Invert the mask
    Mat ncm = (corrMask(hullBB) == 0);

    /*namedWindow("Inverted keypoint mask", WINDOW_AUTOSIZE);
    imshow("Inverted keypoint mask", ncm);*/

    //draw the filled convex hull with enlarged borders
    Mat hullMat1 = Mat::zeros(imgSize, CV_8UC1);
    //with filled contour:
    drawContours(hullMat1, hull, -1, Scalar(255), FILLED);

    /*namedWindow("Convex hull filled", WINDOW_AUTOSIZE);
    imshow("Convex hull filled", hullMat1);*/

    //enlarge borders:
    drawContours(hullMat1, hull, -1, Scalar(255), sqrSi);
    hullMat1 = hullMat1(hullBB);

    /*namedWindow("Convex hull filled enlarged", WINDOW_AUTOSIZE);
    imshow("Convex hull filled enlarged", hullMat1);*/

    //Combine convex hull and inverted mask
    Mat icm = ncm & hullMat1;

    /*namedWindow("Convex hull combined", WINDOW_AUTOSIZE);
    imshow("Convex hull combined", icm);*/

    //Apply distance transform algorithm
    Mat distImg;
    distanceTransform(icm, distImg, DIST_L2, DIST_MASK_PRECISE, CV_32FC1);

    /*Mat distTransNorm;
    normalize(distImg, distTransNorm, 0, 1.0, NORM_MINMAX);
    namedWindow("Distance Transform", WINDOW_AUTOSIZE);
    imshow("Distance Transform", distTransNorm);
    waitKey(0);
    destroyWindow("Distance Transform");*/

    /*destroyWindow("Convex hull combined");
    destroyWindow("Convex hull filled enlarged");
    destroyWindow("Convex hull filled");*/

    //Get the largest distance from white pixels to black pixels
    double minVal, maxVal;
    minMaxLoc(distImg, &minVal, &maxVal);

    //Calculate the kernel size for closing
    int kSize = (int) ceil(maxVal) * 2 + 1;

    //Get structering element
    Mat element = getStructuringElement(MORPH_RECT, Size(kSize, kSize));

    //Perform closing to generate the final mask
    morphologyEx(corrMask, finalMask, MORPH_CLOSE, element);

    /*namedWindow("Final mask for given points", WINDOW_AUTOSIZE);
    imshow("Final mask for given points", finalMask > 0);
    waitKey(0);
    destroyWindow("Final mask for given points");*/
//    destroyWindow("Inverted keypoint mask");
}

//Calculate the seeding position and area for every new moving object
//This function should not be called for the first frame
bool genStereoSequ::getSeedsAreasMovObj() {
    //Get number of possible seeds per region
    Mat nrPerRegMax = Mat::zeros(3, 3, CV_8UC1);
    int minOArea23 = 2 * min(minOArea, maxOArea - minOArea) / 3;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (startPosMovObjs.at<unsigned char>(y, x) == 0)
                continue;
            if(actFracUseableTPperRegion.at<double>(y,x) <= actFracUseableTPperRegionTH)
                continue;
            int moarea = countNonZero(movObjMaskFromLast(regROIs[y][x]) | (actStereoImgsOverlapMask(regROIs[y][x]) == 0));
            if (moarea < minOArea23) {
                nrPerRegMax.at<unsigned char>(y, x) = (unsigned char) maxOPerReg;
            } else if ((moarea <= maxOArea) && (maxOPerReg > 1)) {
                nrPerRegMax.at<unsigned char>(y, x) = (unsigned char) (maxOPerReg - 1);
            } else if (moarea <= (maxOPerReg - 1) * maxOArea) {
                Mat actUsedAreaLabel;
//                int nrLabels = connectedComponents(movObjMaskFromLast(regROIs[y][x]), actUsedAreaLabel, 4, CV_16U);
                Mat actUsedAreaStats;
                Mat actUsedAreaCentroids;
                Mat usedPartMask = movObjMaskFromLast(regROIs[y][x]);
                Mat markSameObj = Mat::zeros(regROIs[y][x].height, regROIs[y][x].width, CV_8UC1);
                int nrLabels = connectedComponentsWithStats(usedPartMask, actUsedAreaLabel, actUsedAreaStats,
                                                            actUsedAreaCentroids, 8, CV_16U);
                int nrFndObj = nrLabels;
                for (int i = 0; i < nrLabels; i++) {
                    Rect labelBB = Rect(actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_LEFT),
                                        actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_TOP),
                                        actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_WIDTH),
                                        actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_HEIGHT));
                    Mat labelPart = actUsedAreaLabel(labelBB);
                    Mat usedPartofPartMask = usedPartMask(labelBB);
                    Point checkPos = Point(0, 0);
                    bool noMovObj = false, validPT = false;
                    for (int y1 = 0; y1 < labelBB.height; ++y1) {
                        for (int x1 = 0; x1 < labelBB.width; ++x1) {
                            if (labelPart.at<uint16_t>(y1, x1) == i) {
                                if (usedPartofPartMask.at<unsigned char>(y1, x1) == 0) {
                                    noMovObj = true;
                                    nrFndObj--;
                                    break;
                                }
                                checkPos = Point(y1, x1);
                                validPT = true;
                                break;
                            }
                        }
                        if (noMovObj || validPT) break;
                    }
                    if (!noMovObj && validPT) {
                        if (markSameObj.at<unsigned char>(checkPos.y + labelBB.y, checkPos.x + labelBB.x) != 0) {
                            nrFndObj--;
                        } else {
                            unsigned char m_obj_nr = usedPartofPartMask.at<unsigned char>(checkPos);
                            markSameObj |= (usedPartMask == m_obj_nr);
                        }
                    }
                }

                if (nrFndObj < maxOPerReg)
                    nrPerRegMax.at<unsigned char>(y, x) = (unsigned char) (maxOPerReg - nrFndObj);
            }
        }
    }

    if (cv::sum(nrPerRegMax)[0] == 0) return false;

    //Get the number of moving object seeds per region
    int nrMovObjs_tmp = (int) pars.nrMovObjs - (int) movObj3DPtsWorld.size();
    Mat nrPerReg = Mat::zeros(3, 3, CV_8UC1);
    while (nrMovObjs_tmp > 0) {
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                if (nrPerRegMax.at<unsigned char>(y, x) > 0) {
                    int addit = (int)(rand2() % 2);
                    if (addit) {
                        nrPerReg.at<unsigned char>(y, x)++;
                        nrPerRegMax.at<unsigned char>(y, x)--;
                        nrMovObjs_tmp--;
                        if (nrMovObjs_tmp == 0)
                            break;
                    }
                }
            }
            if (nrMovObjs_tmp == 0)
                break;
        }
        if (cv::sum(nrPerRegMax)[0] == 0) break;
    }

    //Get the area for each moving object
    std::uniform_int_distribution<int32_t> distribution((int32_t) minOArea, (int32_t) maxOArea);
    movObjAreas.clear();
    movObjAreas = vector<vector<vector<int32_t>>>(3, vector<vector<int32_t>>(3));
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            int nr_tmp = (int) nrPerReg.at<unsigned char>(y, x);
            for (int i = 0; i < nr_tmp; i++) {
                movObjAreas[y][x].push_back(distribution(rand_gen));
            }
        }
    }

    //Get seed positions
    std::vector<std::vector<std::pair<bool,cv::Rect>>> validRects;
    getValidImgRegBorders(actStereoImgsOverlapMask, validRects);
    minODist = imgSize.height / (3 * (maxOPerReg + 1));
    movObjSeeds.clear();
    movObjSeeds = vector<vector<vector<cv::Point_<int32_t>>>>(3, vector<vector<cv::Point_<int32_t>>>(3));
    int maxIt = 20;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            int nr_tmp = (int) nrPerReg.at<unsigned char>(y, x);
            if (nr_tmp > 0) {
                if(!validRects[y][x].first){
                    nrPerReg.at<unsigned char>(y, x) = 0;
                    movObjAreas[y][x].clear();
                    continue;
                }
                rand_gen = std::default_random_engine(
                        rand2());//Prevent getting the same starting positions for equal ranges
                std::uniform_int_distribution<int> distributionX(validRects[y][x].second.x,
                                                                 validRects[y][x].second.x + validRects[y][x].second.width - 1);
                std::uniform_int_distribution<int> distributionY(validRects[y][x].second.y,
                                                                 validRects[y][x].second.y + validRects[y][x].second.height - 1);
                cv::Point_<int32_t> pt;
                int cnt = 0;
                while (cnt < maxIt) {
                    pt = cv::Point_<int32_t>(distributionX(rand_gen), distributionY(rand_gen));
                    if ((movObjMaskFromLast.at<unsigned char>(pt) == 0) && (actStereoImgsOverlapMask.at<unsigned char>(pt) > 0)) {
                        break;
                    }
                    cnt++;
                }
                if (cnt == maxIt) {
                    movObjAreas[y][x].clear();
                    nrPerReg.at<unsigned char>(y, x) = 0;
                    break;
                }
                movObjSeeds[y][x].push_back(pt);
                nr_tmp--;
                if (nr_tmp > 0) {
                    vector<int> xposes, yposes;
                    xposes.push_back(movObjSeeds[y][x].back().x);
                    yposes.push_back(movObjSeeds[y][x].back().y);
                    while (nr_tmp > 0) {
                        vector<double> xInterVals, yInterVals;
                        vector<double> xWeights, yWeights;
                        buildDistributionRanges(xposes, yposes, x, y, xInterVals, xWeights, yInterVals, yWeights, &validRects);

                        //Create piecewise uniform distribution and get a random seed
                        piecewise_constant_distribution<double> distrPieceX(xInterVals.begin(), xInterVals.end(),
                                                                            xWeights.begin());
                        piecewise_constant_distribution<double> distrPieceY(yInterVals.begin(), yInterVals.end(),
                                                                            yWeights.begin());

                        cnt = 0;
                        while (cnt < maxIt) {
                            pt = cv::Point_<int32_t>((int32_t) floor(distrPieceX(rand_gen)),
                                                     (int32_t) floor(distrPieceY(rand_gen)));
                            if ((movObjMaskFromLast.at<unsigned char>(pt) == 0) && (actStereoImgsOverlapMask.at<unsigned char>(pt) > 0)) {
                                break;
                            }
                            cnt++;
                        }
                        if (cnt == maxIt) {
                            for (size_t i = 0; i < (movObjAreas[y][x].size() - movObjSeeds[y][x].size()); i++) {
                                movObjAreas[y][x].pop_back();
                            }
                            nrPerReg.at<unsigned char>(y, x) = (unsigned char) movObjSeeds[y][x].size();
                            break;
                        }

                        movObjSeeds[y][x].push_back(pt);
                        xposes.push_back(movObjSeeds[y][x].back().x);
                        yposes.push_back(movObjSeeds[y][x].back().y);
                        nr_tmp--;
                    }
                }
            }
        }
    }

    return true;
}

//Extracts the areas and seeding positions for new moving objects from the region structure
bool genStereoSequ::getSeedAreaListFromReg(std::vector<cv::Point_<int32_t>> &seeds, std::vector<int32_t> &areas) {
    seeds.clear();
    areas.clear();
    seeds.reserve(pars.nrMovObjs);
    areas.reserve(pars.nrMovObjs);
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (!movObjAreas[y][x].empty()) {
                //copy(movObjAreas[y][x].begin(), movObjAreas[y][x].end(), areas.end());
                areas.insert(areas.end(), movObjAreas[y][x].begin(), movObjAreas[y][x].end());
            }
            if (!movObjSeeds[y][x].empty()) {
                //copy(movObjSeeds[y][x].begin(), movObjSeeds[y][x].end(), seeds.end());
                seeds.insert(seeds.end(), movObjSeeds[y][x].begin(), movObjSeeds[y][x].end());
            }
        }
    }

    CV_Assert(seeds.size() == areas.size());
    if (seeds.empty())
        return false;

    return true;
}

//Check, if it is necessary to calculate new moving objects
//This function should not be called during the first frame
//The function backProjectMovObj() must be called before
bool genStereoSequ::getNewMovObjs() {
    if (pars.minNrMovObjs == 0) {
        clearNewMovObjVars();
        return false;
    }
    if (movObj3DPtsWorld.size() >= pars.minNrMovObjs) {
        clearNewMovObjVars();
        return false;
    }

    return true;
}

//Combines correspondences from static and moving objects
void genStereoSequ::combineCorrespondences() {
    //Get number of TP correspondences
    finalNrTPStatCorrs = actCorrsImg1TP.cols;
    finalNrTPStatCorrsFromLast = actCorrsImg1TPFromLast.cols;
    finalNrTPMovCorrs = 0;
    finalNrTPMovCorrsFromLast = 0;
    combNrCorrsTP = finalNrTPStatCorrsFromLast + finalNrTPStatCorrs;
    for (auto& i : movObjCorrsImg1TPFromLast) {
        if (!i.empty()) {
            combNrCorrsTP += i.cols;
            finalNrTPMovCorrsFromLast += i.cols;
        }
    }
    for (auto& i : movObjCorrsImg1TP) {
        if (!i.empty()) {
            combNrCorrsTP += i.cols;
            finalNrTPMovCorrs += i.cols;
        }
    }
    //Get number of TN correspondences
    combNrCorrsTN = actCorrsImg1TN.cols;
    for (auto& i : movObjCorrsImg1TNFromLast) {
        if (!i.empty()) {
            combNrCorrsTN += i.cols;
        }
    }
    for (auto& i : movObjCorrsImg1TN) {
        if (!i.empty()) {
            combNrCorrsTN += i.cols;
        }
    }

    combCorrsImg1TP.release();
    combCorrsImg2TP.release();
    if (combNrCorrsTP) {
        combCorrsImg1TP = Mat(3, combNrCorrsTP, CV_64FC1);
        combCorrsImg2TP = Mat(3, combNrCorrsTP, CV_64FC1);
    }
    comb3DPts.clear();
    comb3DPts.reserve((size_t)combNrCorrsTP);

    //Copy all TP keypoints of first image
    int actColNr = 0;
    int actColNr2 = actCorrsImg1TPFromLast.cols;
    if (actColNr2 != actColNr) {
        actCorrsImg1TPFromLast.copyTo(combCorrsImg1TP.colRange(actColNr, actColNr2));
    }
    actColNr = actColNr2;
    actColNr2 = actColNr + actCorrsImg1TP.cols;
    if (actColNr2 != actColNr) {
        actCorrsImg1TP.copyTo(combCorrsImg1TP.colRange(actColNr, actColNr2));
    }
    for (auto& i : movObjCorrsImg1TPFromLast) {
        actColNr = actColNr2;
        actColNr2 = actColNr + i.cols;
        if (actColNr2 != actColNr) {
            i.copyTo(combCorrsImg1TP.colRange(actColNr, actColNr2));
        }
    }
    for (auto& i : movObjCorrsImg1TP) {
        actColNr = actColNr2;
        actColNr2 = actColNr + i.cols;
        if (actColNr2 != actColNr) {
            i.copyTo(combCorrsImg1TP.colRange(actColNr, actColNr2));
        }
    }

    //Copy all 3D points
    for (auto& i : actCorrsImg12TPFromLast_Idx) {
        comb3DPts.push_back(actImgPointCloudFromLast[i]);
    }
    if (!actImgPointCloud.empty()) {
        comb3DPts.insert(comb3DPts.end(), actImgPointCloud.begin(), actImgPointCloud.end());
    }
    for (size_t i = 0; i < movObjCorrsImg12TPFromLast_Idx.size(); i++) {
        for (auto& j : movObjCorrsImg12TPFromLast_Idx[i]) {
            comb3DPts.push_back(movObj3DPtsCam[i][j]);
        }
    }
    for (auto& i : movObj3DPtsCamNew) {
        //copy(i.begin(), i.end(), comb3DPts.end());
        if (!i.empty()) {
            comb3DPts.insert(comb3DPts.end(), i.begin(), i.end());
        }
    }

    CV_Assert(combCorrsImg1TP.cols == (int) comb3DPts.size());

    //Copy all TP keypoints of second image
    actColNr = 0;
    actColNr2 = actCorrsImg2TPFromLast.cols;
    if (actColNr2 != actColNr) {
        actCorrsImg2TPFromLast.copyTo(combCorrsImg2TP.colRange(actColNr, actColNr2));
    }
    actColNr = actColNr2;
    actColNr2 = actColNr + actCorrsImg2TP.cols;
    if (actColNr2 != actColNr) {
        actCorrsImg2TP.copyTo(combCorrsImg2TP.colRange(actColNr, actColNr2));
    }
    for (auto& i : movObjCorrsImg2TPFromLast) {
        actColNr = actColNr2;
        actColNr2 = actColNr + i.cols;
        if (actColNr2 != actColNr) {
            i.copyTo(combCorrsImg2TP.colRange(actColNr, actColNr2));
        }
    }
    for (auto& i : movObjCorrsImg2TP) {
        actColNr = actColNr2;
        actColNr2 = actColNr + i.cols;
        if (actColNr2 != actColNr) {
            i.copyTo(combCorrsImg2TP.colRange(actColNr, actColNr2));
        }
    }

    combCorrsImg1TN.release();
    combCorrsImg2TN.release();
    if (combNrCorrsTN) {
        combCorrsImg1TN = Mat(3, combNrCorrsTN, CV_64FC1);
        combCorrsImg2TN = Mat(3, combNrCorrsTN, CV_64FC1);
    }

    //Copy all TN keypoints of first image
    finalNrTNStatCorrs = actCorrsImg1TN.cols;
    finalNrTNMovCorrs = 0;
    actColNr = 0;
    actColNr2 = finalNrTNStatCorrs;
    if (actColNr2 != actColNr) {
        actCorrsImg1TN.copyTo(combCorrsImg1TN.colRange(actColNr, actColNr2));
    }
    for (auto& i : movObjCorrsImg1TNFromLast) {
        actColNr = actColNr2;
        actColNr2 = actColNr + i.cols;
        if (actColNr2 != actColNr) {
            i.copyTo(combCorrsImg1TN.colRange(actColNr, actColNr2));
            finalNrTNMovCorrs += i.cols;
        }
    }
    for (auto& i : movObjCorrsImg1TN) {
        actColNr = actColNr2;
        actColNr2 = actColNr + i.cols;
        if (actColNr2 != actColNr) {
            i.copyTo(combCorrsImg1TN.colRange(actColNr, actColNr2));
            finalNrTNMovCorrs += i.cols;
        }
    }

    //Copy all TN keypoints of second image
    actColNr = 0;
    actColNr2 = actCorrsImg2TN.cols;
    if (actColNr2 != actColNr) {
        actCorrsImg2TN.copyTo(combCorrsImg2TN.colRange(actColNr, actColNr2));
    }
    for (auto& i : movObjCorrsImg2TNFromLast) {
        actColNr = actColNr2;
        actColNr2 = actColNr + i.cols;
        if (actColNr2 != actColNr) {
            i.copyTo(combCorrsImg2TN.colRange(actColNr, actColNr2));
        }
    }
    for (auto& i : movObjCorrsImg2TN) {
        actColNr = actColNr2;
        actColNr2 = actColNr + i.cols;
        if (actColNr2 != actColNr) {
            i.copyTo(combCorrsImg2TN.colRange(actColNr, actColNr2));
        }
    }

    //Copy distances of TN locations to their real matching position
    combDistTNtoReal.clear();
    combDistTNtoReal.reserve((size_t)combNrCorrsTN);
    if (!distTNtoReal.empty()) {
        combDistTNtoReal.insert(combDistTNtoReal.end(), distTNtoReal.begin(), distTNtoReal.end());
    }
    for (auto& i : movObjDistTNtoReal) {
        //copy(i.begin(), i.end(), combDistTNtoReal.end());
        if (!i.empty()) {
            combDistTNtoReal.insert(combDistTNtoReal.end(), i.begin(), i.end());
        }
    }
    for (auto& i : movObjDistTNtoRealNew) {
        //copy(i.begin(), i.end(), combDistTNtoReal.end());
        if (!i.empty()) {
            combDistTNtoReal.insert(combDistTNtoReal.end(), i.begin(), i.end());
        }
    }

    CV_Assert((size_t) combCorrsImg1TN.cols == combDistTNtoReal.size());

    if(verbose & SHOW_COMBINED_CORRESPONDENCES){
        visualizeAllCorrespondences();
    }

    //Check global inlier ratio of backprojected and new static and moving objects
    if(verbose & PRINT_WARNING_MESSAGES) {
        double inlRatDiffSR = (double) combNrCorrsTP / (double) (combNrCorrsTP + combNrCorrsTN) - inlRat[actFrameCnt];
        double testVal = min((double) (combNrCorrsTP + combNrCorrsTN) / 100.0, 1.0) * inlRatDiffSR / 300.0;
        if (!nearZero(testVal)) {
            cout
                    << "Inlier ratio of combined static and moving correspondences differs from global inlier ratio (0 - 1.0) by "
                    << inlRatDiffSR << endl;
        }
    }
}

void genStereoSequ::combineWorldCoordinateIndices(){
    combCorrsImg12TP_IdxWorld.clear();
    combCorrsImg12TP_IdxWorld2.clear();

    //Copy all 3D world coordinate indices
    for (auto& i : actCorrsImg12TPFromLast_Idx) {
        combCorrsImg12TP_IdxWorld.push_back(static_cast<int64_t>(actCorrsImg12TPFromLast_IdxWorld[i]));
    }
    if (!actImgPointCloud.empty()) {
        CV_Assert(!actCorrsImg12TP_IdxWorld.empty());
        combCorrsImg12TP_IdxWorld.reserve(combCorrsImg12TP_IdxWorld.size() + actCorrsImg12TP_IdxWorld.size());
        for(auto& i : actCorrsImg12TP_IdxWorld){
            combCorrsImg12TP_IdxWorld.push_back(static_cast<int64_t>(i));
        }
    }
    combCorrsImg12TPContMovObj_IdxWorld = combCorrsImg12TP_IdxWorld;
    combCorrsImg12TP_IdxWorld2 = combCorrsImg12TP_IdxWorld;

    int64_t idxMOAllFrames = static_cast<int64_t>(movObj3DPtsWorldAllFrames.size()) -
            static_cast<int64_t>(movObjCorrsImg12TPFromLast_Idx.size()) -
            static_cast<int64_t>(actCorrsOnMovObj_IdxWorld.size());
    nrMovingObjPerFrame.push_back(movObjCorrsImg12TPFromLast_Idx.size() + actCorrsOnMovObj_IdxWorld.size());
    for (size_t i = 0; i < movObjCorrsImg12TPFromLast_Idx.size(); i++) {
        combCorrsImg12TP_IdxWorld.reserve(combCorrsImg12TP_IdxWorld.size() + movObjCorrsImg12TPFromLast_Idx[i].size());
        combCorrsImg12TP_IdxWorld2.reserve(combCorrsImg12TP_IdxWorld2.size() + movObjCorrsImg12TPFromLast_Idx[i].size());
        combCorrsImg12TPContMovObj_IdxWorld.reserve(combCorrsImg12TPContMovObj_IdxWorld.size() +
        movObjCorrsImg12TPFromLast_Idx[i].size());
        const auto idxMO = static_cast<int64_t>(i + 1);
        const int64_t idxMOAll = idxMO + idxMOAllFrames;
        const auto idx_FrEmerge = static_cast<int64_t>(get<0>(movObjFrameEmerge[i])) << 8;
        const auto init_idx = static_cast<int64_t>(get<1>(movObjFrameEmerge[i]) + 1);
        for (auto& j : movObjCorrsImg12TPFromLast_Idx[i]) {
            auto idxPts = static_cast<int64_t>(actCorrsOnMovObjFromLast_IdxWorld[i][j]);
            auto idx_emerge = static_cast<int64_t>(movObjImg12TP_InitIdxWorld[i][idxPts]);
            CV_Assert(idxPts < static_cast<int64_t>(INT32_MAX));
            idxPts = idxPts << 32;
            idx_emerge = (idx_emerge + 1) << 32;
            int64_t idx = -1 * (idxMO | idxPts);
            combCorrsImg12TP_IdxWorld.push_back(idx);
            idx = -1 * (idxMOAll | idxPts);
            combCorrsImg12TPContMovObj_IdxWorld.push_back(idx);
            const auto idx_id = -1 * (idx_emerge | idx_FrEmerge | init_idx);
            combCorrsImg12TP_IdxWorld2.push_back(idx_id);
        }
    }
    const auto movObjFromLastSi = static_cast<int64_t>(movObjCorrsImg12TPFromLast_Idx.size());
    idxMOAllFrames += movObjFromLastSi;
    for (size_t i = 0; i < actCorrsOnMovObj_IdxWorld.size(); i++) {
        if (!actCorrsOnMovObj_IdxWorld[i].empty()) {
            combCorrsImg12TP_IdxWorld.reserve(combCorrsImg12TP_IdxWorld.size() + actCorrsOnMovObj_IdxWorld[i].size());
            combCorrsImg12TP_IdxWorld2.reserve(combCorrsImg12TP_IdxWorld2.size() + actCorrsOnMovObj_IdxWorld[i].size());
            combCorrsImg12TPContMovObj_IdxWorld.reserve(combCorrsImg12TPContMovObj_IdxWorld.size() +
            actCorrsOnMovObj_IdxWorld[i].size());
            const auto idxMO = static_cast<int64_t>(i + 1);
            const auto idx2 = static_cast<size_t>(movObjFromLastSi) + i;
            const int64_t idxMOAll = idxMO + idxMOAllFrames;
            get<1>(movObjFrameEmerge[i]) = i;
            const auto idx_frCnt = static_cast<int64_t>(actFrameCnt) << 8;
            for (auto& j : actCorrsOnMovObj_IdxWorld[i]) {
                auto idxPts = static_cast<int64_t>(j);
                auto idx_emerge = static_cast<int64_t>(movObjImg12TP_InitIdxWorld[idx2][j]);
                CV_Assert(idxPts < static_cast<int64_t>(INT32_MAX));
                idxPts = idxPts << 32;
                idx_emerge = (idx_emerge + 1) << 32;
                int64_t idx = -1 * (idxMO | idxPts);
                combCorrsImg12TP_IdxWorld.push_back(idx);
                idx = -1 * (idxMOAll | idxPts);
                combCorrsImg12TPContMovObj_IdxWorld.push_back(idx);
                const auto idx_id = -1 * (idx_emerge | idx_frCnt | (idxMO + movObjFromLastSi));
                combCorrsImg12TP_IdxWorld2.push_back(idx_id);
            }
        }
    }
}

void genStereoSequ::visualizeAllCorrespondences(){
    Mat allCorrs = cv::Mat::zeros(imgSize, CV_8UC3);

    for (int i = 0; i < actCorrsImg1TPFromLast.cols; ++i) {
        int x = (int)round(actCorrsImg1TPFromLast.at<double>(0,i));
        int y = (int)round(actCorrsImg1TPFromLast.at<double>(1,i));
        allCorrs.at<cv::Vec3b>(y,x)[1] = 255;
    }
    int cnt_overlaps = 0;
    for (int i = 0; i < actCorrsImg1TP.cols; ++i) {
        int x = (int)round(actCorrsImg1TP.at<double>(0,i));
        int y = (int)round(actCorrsImg1TP.at<double>(1,i));
        if((allCorrs.at<cv::Vec3b>(y,x)[0] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[1] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[2] == 0)){
            allCorrs.at<cv::Vec3b>(y,x)[1] = 255;
        }
        else{
            allCorrs.at<cv::Vec3b>(y,x)[1] = 0;
            allCorrs.at<cv::Vec3b>(y,x)[2] = 255;
            cnt_overlaps++;
        }
    }
    for (auto &j : movObjCorrsImg1TPFromLast) {
        for (int i = 0; i < j.cols; ++i) {
            int x = (int)round(j.at<double>(0,i));
            int y = (int)round(j.at<double>(1,i));
            if((allCorrs.at<cv::Vec3b>(y,x)[0] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[1] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[2] == 0)){
                allCorrs.at<cv::Vec3b>(y,x)[0] = 255;
            }
            else{
                allCorrs.at<cv::Vec3b>(y,x)[2] = 255;
                cnt_overlaps++;
            }
        }
    }
    for (auto &j : movObjCorrsImg1TP) {
        for (int i = 0; i < j.cols; ++i) {
            int x = (int)round(j.at<double>(0,i));
            int y = (int)round(j.at<double>(1,i));
            if((allCorrs.at<cv::Vec3b>(y,x)[0] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[1] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[2] == 0)){
                allCorrs.at<cv::Vec3b>(y,x)[0] = 255;
            }
            else{
                allCorrs.at<cv::Vec3b>(y,x)[2] = 255;
                cnt_overlaps++;
            }
        }
    }

    for (int i = 0; i < actCorrsImg1TN.cols; ++i) {
        int x = (int)round(actCorrsImg1TN.at<double>(0,i));
        int y = (int)round(actCorrsImg1TN.at<double>(1,i));
        if((allCorrs.at<cv::Vec3b>(y,x)[0] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[1] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[2] == 0)){
            allCorrs.at<cv::Vec3b>(y,x)[1] = 127;
        }
        else{
            allCorrs.at<cv::Vec3b>(y,x)[1] = 0;
            allCorrs.at<cv::Vec3b>(y,x)[2] = 127;
            cnt_overlaps++;
        }
    }

    for (auto &j : movObjCorrsImg1TNFromLast) {
        for (int i = 0; i < j.cols; ++i) {
            int x = (int)round(j.at<double>(0,i));
            int y = (int)round(j.at<double>(1,i));
            if((allCorrs.at<cv::Vec3b>(y,x)[0] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[1] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[2] == 0)){
                allCorrs.at<cv::Vec3b>(y,x)[0] = 127;
            }
            else{
                allCorrs.at<cv::Vec3b>(y,x)[2] = 127;
                cnt_overlaps++;
            }
        }
    }
    for (auto &j : movObjCorrsImg1TN) {
        for (int i = 0; i < j.cols; ++i) {
            int x = (int)round(j.at<double>(0,i));
            int y = (int)round(j.at<double>(1,i));
            if((allCorrs.at<cv::Vec3b>(y,x)[0] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[1] == 0) && (allCorrs.at<cv::Vec3b>(y,x)[2] == 0)){
                allCorrs.at<cv::Vec3b>(y,x)[0] = 127;
            }
            else{
                allCorrs.at<cv::Vec3b>(y,x)[2] = 127;
                cnt_overlaps++;
            }
        }
    }

    if(cnt_overlaps > 0){
        cout << "Found " << cnt_overlaps << " overlapping correspondences!" << endl;
    }
    if(!writeIntermediateImg(allCorrs, "combined_correspondences_in_image_1")) {
        namedWindow("Combined correspondences in image 1", WINDOW_AUTOSIZE);
        imshow("Combined correspondences in image 1", allCorrs);
        waitKey(0);
        destroyWindow("Combined correspondences in image 1");
    }
}

//Get the paramters and indices for the actual frame. This function must be called before simulating a new stereo frame
void genStereoSequ::updateFrameParameters() {
    if (((actFrameCnt % (pars.nFramesPerCamConf)) == 0) && (actFrameCnt > 0)) {
        actStereoCIdx++;
    }
    actR = R[actStereoCIdx];
    actT = t[actStereoCIdx];
    actKd1 = K1_distorted[actStereoCIdx];
    actKd2 = K2_distorted[actStereoCIdx];
    actDepthNear = depthNear[actStereoCIdx];
    actDepthMid = depthMid[actStereoCIdx];
    actDepthFar = depthFar[actStereoCIdx];
    actFracUseableTPperRegion = fracUseableTPperRegion[actStereoCIdx];
    actStereoImgsOverlapMask = stereoImgsOverlapMask[actStereoCIdx];

    if (((actFrameCnt % (pars.corrsPerRegRepRate)) == 0) && (actFrameCnt > 0)) {
        actCorrsPRIdx++;
        if (actCorrsPRIdx >= pars.corrsPerRegion.size()) {
            actCorrsPRIdx = 0;
        }
    }
}

//Insert new generated 3D points into the world coordinate point cloud
void genStereoSequ::transPtsToWorld() {
    if (actImgPointCloud.empty()) {
        return;
    }

    size_t nrPts = actImgPointCloud.size();
    size_t nrOldPts = staticWorld3DPts->size();

    actCorrsImg12TP_IdxWorld.clear();
    actCorrsImg12TP_IdxWorld.resize(nrPts);
    std::iota(actCorrsImg12TP_IdxWorld.begin(), actCorrsImg12TP_IdxWorld.end(), nrOldPts);

    staticWorld3DPts->reserve(nrOldPts + nrPts);

    for (size_t i = 0; i < nrPts; i++) {
        Mat ptm = absCamCoordinates[actFrameCnt].R * Mat(actImgPointCloud[i]).reshape(1) +
                  absCamCoordinates[actFrameCnt].t;
        staticWorld3DPts->push_back(
                pcl::PointXYZ((float) ptm.at<double>(0), (float) ptm.at<double>(1), (float) ptm.at<double>(2)));
    }

    if (verbose & SHOW_STATIC_OBJ_3D_PTS) {
        visualizeStaticObjPtCloud();
    }
    destroyAllWindows();

    if ((verbose & SHOW_MOV_OBJ_3D_PTS) && (verbose & SHOW_STATIC_OBJ_3D_PTS)) {
        visualizeMovingAndStaticObjPtCloud();
    }
}

//Transform new generated 3D points from new generated moving objects into world coordinates
void genStereoSequ::transMovObjPtsToWorld() {
    if (movObj3DPtsCamNew.empty()) {
        return;
    }

    size_t nrNewObjs = movObj3DPtsCamNew.size();
    size_t nrOldObjs = movObj3DPtsWorld.size();
    size_t nrSaveObjs = movObj3DPtsWorldAllFrames.size();
    movObj3DPtsWorld.resize(nrOldObjs + nrNewObjs);
    movObjFrameEmerge.resize(nrOldObjs + nrNewObjs);
    movObjImg12TP_InitIdxWorld.resize(nrOldObjs + nrNewObjs);
    movObjWorldMovement.resize(nrOldObjs + nrNewObjs);
    movObj3DPtsWorldAllFrames.resize(nrSaveObjs + nrNewObjs);
    //Add new depth classes to existing ones
    movObjDepthClass.insert(movObjDepthClass.end(), movObjDepthClassNew.begin(), movObjDepthClassNew.end());
    /* WRONG:
     * Mat trans_c2w;//Translation vector for transferring 3D points from camera to world coordinates
    Mat RC2W = absCamCoordinates[actFrameCnt].R.t();
    trans_c2w = -1.0 * RC2W * absCamCoordinates[actFrameCnt].t;//Get the C2W-translation from the position of the camera centre in world coordinates*/
    for (size_t i = 0; i < nrNewObjs; i++) {
        size_t idx = nrOldObjs + i;
        movObj3DPtsWorld[idx].reserve(movObj3DPtsCamNew[i].size());
        movObjImg12TP_InitIdxWorld[idx].resize(movObj3DPtsCamNew[i].size());
        std::iota (movObjImg12TP_InitIdxWorld[idx].begin(), movObjImg12TP_InitIdxWorld[idx].end(), 0);
        for (size_t j = 0; j < movObj3DPtsCamNew[i].size(); j++) {
            Mat pt3 = Mat(movObj3DPtsCamNew[i][j]).reshape(1);
            //X_world  = R * X_cam + t
            Mat ptm = absCamCoordinates[actFrameCnt].R * pt3 + absCamCoordinates[actFrameCnt].t;
//            Mat ptm = RC2W * pt3 + trans_c2w;
            movObj3DPtsWorld[idx].push_back(
                    pcl::PointXYZ((float) ptm.at<double>(0), (float) ptm.at<double>(1), (float) ptm.at<double>(2)));
        }
        movObj3DPtsWorldAllFrames[nrSaveObjs + i] = movObj3DPtsWorld[idx];
        double velocity = 0;
        if (nearZero(pars.relMovObjVelRange.first - pars.relMovObjVelRange.second)) {
            velocity = absCamVelocity * pars.relMovObjVelRange.first;
        } else {
            double relV = getRandDoubleValRng(pars.relMovObjVelRange.first, pars.relMovObjVelRange.second);
            velocity = absCamVelocity * relV;
        }
        Mat tdiff;
        if ((actFrameCnt + 1) < totalNrFrames) {
            //Get direction of camera from actual to next frame
            tdiff = absCamCoordinates[actFrameCnt + 1].t - absCamCoordinates[actFrameCnt].t;
        } else {
            //Get direction of camera from last to actual frame
            tdiff = absCamCoordinates[actFrameCnt].t - absCamCoordinates[actFrameCnt - 1].t;
        }
        double tnorm = norm(tdiff);
        if(nearZero(tnorm))
            tdiff = Mat::zeros(3,1,CV_64FC1);
        else
            tdiff /= tnorm;
        //Add the movement direction of the moving object
        tdiff += movObjDir;
        tnorm = norm(tdiff);
        if(nearZero(tnorm))
            tdiff = Mat::zeros(3,1,CV_64FC1);
        else
            tdiff /= tnorm;
        tdiff *= velocity;
        movObjWorldMovement[idx] = tdiff.clone();
        movObjFrameEmerge[idx] = make_tuple(actFrameCnt, 0, tdiff.clone());
    }

    if (verbose & SHOW_MOV_OBJ_3D_PTS) {
        visualizeMovObjPtCloud();
    }
    destroyAllWindows();
}

void genStereoSequ::visualizeMovObjPtCloud() {
    if (movObj3DPtsWorld.empty())
        return;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("Moving objects"));

    Eigen::Affine3f m = initPCLViewerCoordinateSystems(viewer, absCamCoordinates[actFrameCnt].R,
                                                       absCamCoordinates[actFrameCnt].t);

    //Generate colormap for moving obejcts (every object has a different color)
    Mat colormap_img;
    getNColors(colormap_img, movObj3DPtsWorld.size(), COLORMAP_PARULA);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr basic_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    int idx = 0;
    for (auto &i : movObj3DPtsWorld) {
        for (auto &j : i) {
            pcl::PointXYZRGB point;
            point.x = j.x;
            point.y = j.y;
            point.z = j.z;

            point.b = colormap_img.at<cv::Vec3b>(idx)[0];
            point.g = colormap_img.at<cv::Vec3b>(idx)[1];
            point.r = colormap_img.at<cv::Vec3b>(idx)[2];
            basic_cloud_ptr->push_back(point);
        }
        idx++;
    }
    viewer->addPointCloud<pcl::PointXYZRGB>(basic_cloud_ptr, "moving objects cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
                                             "moving objects cloud");


    setPCLViewerCamPars(viewer, m.matrix(), K1);

    startPCLViewer(viewer);
}


void genStereoSequ::visualizeStaticObjPtCloud() {
    if (staticWorld3DPts->empty())
        return;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("Static Objects"));

    Eigen::Affine3f m = initPCLViewerCoordinateSystems(viewer, absCamCoordinates[actFrameCnt].R,
                                                       absCamCoordinates[actFrameCnt].t);

    viewer->addPointCloud<pcl::PointXYZ>(staticWorld3DPts, "static objects cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
                                             "static objects cloud");

    setPCLViewerCamPars(viewer, m.matrix(), K1);

    startPCLViewer(viewer);
}

void genStereoSequ::visualizeMovingAndStaticObjPtCloud() {
    if (staticWorld3DPts->empty() || movObj3DPtsWorld.empty())
        return;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("Moving and static objects"));

    Eigen::Affine3f m = initPCLViewerCoordinateSystems(viewer, absCamCoordinates[actFrameCnt].R,
                                                       absCamCoordinates[actFrameCnt].t);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr basic_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

    //Generate colormap for moving obejcts (every object has a different color)
    Mat colormap_img;
    getNColors(colormap_img, movObj3DPtsWorld.size(), COLORMAP_AUTUMN);

    int idx = 0;
    for (auto &i : movObj3DPtsWorld) {
        for (auto &j : i) {
            pcl::PointXYZRGB point;
            point.x = j.x;
            point.y = j.y;
            point.z = j.z;

            point.b = colormap_img.at<cv::Vec3b>(idx)[0];
            point.g = colormap_img.at<cv::Vec3b>(idx)[1];
            point.r = colormap_img.at<cv::Vec3b>(idx)[2];
            basic_cloud_ptr->push_back(point);
        }
        idx++;
    }
    for (const auto &i : *(staticWorld3DPts.get())) {
        pcl::PointXYZRGB point;
        point.x = i.x;
        point.y = i.y;
        point.z = i.z;

        point.b = 0;
        point.g = 255;
        point.r = 0;
        basic_cloud_ptr->push_back(point);
    }
    viewer->addPointCloud<pcl::PointXYZRGB>(basic_cloud_ptr, "static and moving objects cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
                                             "static and moving objects cloud");

    setPCLViewerCamPars(viewer, m.matrix(), K1);

    startPCLViewer(viewer);
}

void genStereoSequ::visualizeMovObjMovement(std::vector<pcl::PointXYZ> &cloudCentroids_old,
                                            std::vector<pcl::PointXYZ> &cloudCentroids_new,
                                            std::vector<float> &cloudExtensions) {
    if (cloudCentroids_old.empty() || cloudCentroids_new.empty())
        return;

    CV_Assert(cloudCentroids_old.size() == cloudCentroids_new.size());

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("Moving object movement"));

    Eigen::Affine3f m = initPCLViewerCoordinateSystems(viewer, absCamCoordinates[actFrameCnt].R,
                                                       absCamCoordinates[actFrameCnt].t);

    //Generate colormap for moving obejcts (every object has a different color)
    Mat colormap_img;
    getNColors(colormap_img, cloudCentroids_old.size() * 2, COLORMAP_PARULA);
    colormap_img.convertTo(colormap_img, CV_64FC3);
    colormap_img /= 255.0;

    for (int i = 0; i < (int)cloudCentroids_old.size(); ++i) {
        int i2 = 2 * i;
        int i21 = i2 + 1;
        viewer->addSphere(cloudCentroids_old[i],
                          (double) cloudExtensions[i],
                          colormap_img.at<cv::Vec3d>(i2)[2],
                          colormap_img.at<cv::Vec3d>(i2)[1],
                          colormap_img.at<cv::Vec3d>(i2)[0],
                          "sphere_old" + std::to_string(i));
        viewer->addSphere(cloudCentroids_new[i],
                          (double) cloudExtensions[i],
                          colormap_img.at<cv::Vec3d>(i21)[2],
                          colormap_img.at<cv::Vec3d>(i21)[1],
                          colormap_img.at<cv::Vec3d>(i21)[0],
                          "sphere_new" + std::to_string(i));
        viewer->addArrow(cloudCentroids_new[i],
                         cloudCentroids_old[i],
                         colormap_img.at<cv::Vec3d>(i2)[2],
                         colormap_img.at<cv::Vec3d>(i2)[1],
                         colormap_img.at<cv::Vec3d>(i2)[0],
                         false,
                         "arrow" + std::to_string(i));
    }

    //Add last camera center
    if (actFrameCnt > 0) {
        addVisualizeCamCenter(viewer, absCamCoordinates[actFrameCnt - 1].R, absCamCoordinates[actFrameCnt - 1].t);
        pcl::PointXYZ c_old, c_new;
        c_old.x = (float) absCamCoordinates[actFrameCnt - 1].t.at<double>(0);
        c_old.y = (float) absCamCoordinates[actFrameCnt - 1].t.at<double>(1);
        c_old.z = (float) absCamCoordinates[actFrameCnt - 1].t.at<double>(2);
        c_new.x = (float) absCamCoordinates[actFrameCnt].t.at<double>(0);
        c_new.y = (float) absCamCoordinates[actFrameCnt].t.at<double>(1);
        c_new.z = (float) absCamCoordinates[actFrameCnt].t.at<double>(2);
        viewer->addArrow(c_new,
                         c_old,
                         1.0,
                         1.0,
                         1.0,
                         false,
                         "arrow_cams");
    }

    setPCLViewerCamPars(viewer, m.matrix(), K1);

    startPCLViewer(viewer);
}

//Get the relative movement direction (compared to the camera movement) for every moving object
void genStereoSequ::checkMovObjDirection() {
    if (pars.movObjDir.empty()) {
        Mat newMovObjDir(3, 1, CV_64FC1);
        cv::randu(newMovObjDir, Scalar(0), Scalar(1.0));
        double newMovObjDirNorm = norm(newMovObjDir);
        if(nearZero(newMovObjDirNorm))
            newMovObjDir = Mat::zeros(3,1,CV_64FC1);
        else
            newMovObjDir /= newMovObjDirNorm;
        newMovObjDir.copyTo(movObjDir);
    } else {
        movObjDir = pars.movObjDir;
        double movObjDirNorm = norm(movObjDir);
        if(nearZero(movObjDirNorm))
            movObjDir = Mat::zeros(3,1,CV_64FC1);
        else
            movObjDir /= movObjDirNorm;
    }
}

//Updates the actual position of moving objects and their corresponding 3D points according to their moving direction and velocity.
void genStereoSequ::updateMovObjPositions() {
    if (movObj3DPtsWorld.empty()) {
        return;
    }

    vector<pcl::PointXYZ> mocentroids;
    if (verbose & SHOW_MOV_OBJ_MOVEMENT) {
        getCloudCentroids(movObj3DPtsWorld, mocentroids);
    }

    for (size_t i = 0; i < movObj3DPtsWorld.size(); i++) {
        Eigen::Affine3f obj_transform = Eigen::Affine3f::Identity();
        obj_transform.translation() << (float) movObjWorldMovement[i].at<double>(0),
                (float) movObjWorldMovement[i].at<double>(1),
                (float) movObjWorldMovement[i].at<double>(2);
        pcl::transformPointCloud(movObj3DPtsWorld[i], movObj3DPtsWorld[i], obj_transform);
    }

    if (verbose & SHOW_MOV_OBJ_MOVEMENT) {
        vector<pcl::PointXYZ> mocentroids2;
        vector<float> cloudExtensions;
        getCloudCentroids(movObj3DPtsWorld, mocentroids2);
        getMeanCloudStandardDevs(movObj3DPtsWorld, cloudExtensions, mocentroids2);
        visualizeMovObjMovement(mocentroids, mocentroids2, cloudExtensions);
    }
}

//Get 3D-points of moving objects that are visible in the camera and transform them from the world coordinate system into camera coordinate system
void genStereoSequ::getMovObjPtsCam() {
    if (movObj3DPtsWorld.empty()) {
        return;
    }

    size_t movObjSize = movObj3DPtsWorld.size();
    vector<int> delList;
    movObj3DPtsCam.clear();
    movObj3DPtsCam.resize(movObjSize);
    actCorrsOnMovObjFromLast_IdxWorld.clear();
    actCorrsOnMovObjFromLast_IdxWorld.resize(movObjSize);
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> filteredOccludedPts(movObjSize);
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> filteredOccludedCamPts(movObjSize);
    vector<vector<int>> filteredOccludedCamPts_idx(movObjSize);
    vector<pair<float, int>> meanDists(movObjSize, make_pair(0, -1));

    Eigen::Affine3f obj_transform = Eigen::Affine3f::Identity();
    Eigen::Vector3d te;
    Eigen::Matrix3d Re;
    cv::cv2eigen(absCamCoordinates[actFrameCnt].R.t(), Re);
    cv::cv2eigen(absCamCoordinates[actFrameCnt].t, te);
    te = -1.0 * Re * te;
    obj_transform.matrix().block<3, 3>(0, 0) = Re.cast<float>();
    obj_transform.matrix().block<3, 1>(0, 3) = te.cast<float>();

    for (int i = 0; i < (int)movObjSize; i++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_movObj3DPtsWorld(movObj3DPtsWorld[i].makeShared());
        pcl::PointCloud<pcl::PointXYZ>::Ptr camFilteredPts(new pcl::PointCloud<pcl::PointXYZ>());
        vector<int> camFilteredPts_idx;
        //Check if the moving object is visible in the camera
        bool success = getVisibleCamPointCloud(ptr_movObj3DPtsWorld, &camFilteredPts_idx);
        if (!success) {
            delList.push_back(i);
            continue;
        }

        camFilteredPts->reserve(camFilteredPts_idx.size());
        for(auto& j : camFilteredPts_idx){
            camFilteredPts->push_back(movObj3DPtsWorld[i][j]);
        }

        //Check if due to the changed camera position, some 3D points are occluded from others of the same moving object
        success = filterNotVisiblePts(camFilteredPts, actCorrsOnMovObjFromLast_IdxWorld[i]);
        if (!success) {
            actCorrsOnMovObjFromLast_IdxWorld[i].clear();
            success = filterNotVisiblePts(camFilteredPts, actCorrsOnMovObjFromLast_IdxWorld[i], true);
            if (!success) {
                if (actCorrsOnMovObjFromLast_IdxWorld[i].size() < 5) {
                    delList.push_back(i);
                    continue;
                }
            }
        }

        filteredOccludedPts[i].reset(new pcl::PointCloud<pcl::PointXYZ>());
        for(auto& j : actCorrsOnMovObjFromLast_IdxWorld[i]){
            filteredOccludedPts[i]->push_back(camFilteredPts->at((size_t)j));
            j = camFilteredPts_idx[j];
        }

        //Convert 3D points from world into camera coordinates
        filteredOccludedCamPts[i].reset(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*filteredOccludedPts[i].get(), *filteredOccludedCamPts[i].get(), obj_transform);

        //Get the mean distance of every moving object to the camera
        if (movObjSize > 1) {
            pcl::PointXYZ mocentroid;
            getCloudCentroid(*filteredOccludedCamPts[i].get(), mocentroid);
            meanDists[i] = make_pair(mocentroid.z, i);
        }
    }

    if ((movObjSize - delList.size()) > 1) {
        //Check, if there are overlaps between moving objects and filter 3D points that would be behind another moving object
        sort(meanDists.begin(), meanDists.end(), [](std::pair<float, int> first, std::pair<float, int> second) {
            return first.first < second.first;
        });

        int sqrSi = csurr.rows;
        int posadd = (sqrSi - 1) / 2;
        Mat globMOmask = Mat::zeros(imgSize, CV_8UC1);
        for (size_t i = 0; i < movObjSize; i++) {
            int idx = meanDists[i].second;
            if (idx < 0)
                continue;

//            bool kpOutofR = false;
            vector<int> keyPDelList;
            Mat locMOmask = Mat::zeros(imgSize.height + sqrSi - 1, imgSize.width + sqrSi - 1, CV_8UC1);
            std::vector<cv::Point> keypointsMO(filteredOccludedCamPts[idx]->size());
            for (int j = 0; j < (int)filteredOccludedCamPts[idx]->size(); ++j) {
                Mat pt = K1 * (Mat_<double>(3, 1) << (double) (*filteredOccludedCamPts[idx])[j].x,
                        (double) (*filteredOccludedCamPts[idx])[j].y,
                        (double) (*filteredOccludedCamPts[idx])[j].z);
                if(nearZero(pt.at<double>(2))){
                    keyPDelList.push_back(j);
                    continue;
                }
                pt /= pt.at<double>(2);
                keypointsMO[j].x = (int) round(pt.at<double>(0));
                keypointsMO[j].y = (int) round(pt.at<double>(1));
                if ((keypointsMO[j].x < 0) || (keypointsMO[j].y < 0) ||
                    (keypointsMO[j].x >= imgSize.width) ||
                    (keypointsMO[j].y >= imgSize.height)) {
//                    kpOutofR = true;
                    keyPDelList.push_back(j);
                    continue;
                }
                Mat s_tmp = locMOmask(Rect(keypointsMO[j], Size(sqrSi, sqrSi)));
                csurr.copyTo(s_tmp);
            }
            if (!keyPDelList.empty()) {
                for (int j = (int) keyPDelList.size() - 1; j >= 0; j--) {
                    keypointsMO.erase(keypointsMO.begin() + keyPDelList[j]);
                    filteredOccludedCamPts[idx]->erase(filteredOccludedCamPts[idx]->begin() + keyPDelList[j]);
                    actCorrsOnMovObjFromLast_IdxWorld[idx].erase(actCorrsOnMovObjFromLast_IdxWorld[idx].begin() + keyPDelList[j]);
                }
                if (filteredOccludedCamPts[idx]->empty()) {
                    delList.push_back(idx);
                    continue;
                }
            }

            if (verbose & SHOW_BACKPROJECT_OCCLUSIONS_MOV_OBJ) {
                if(!writeIntermediateImg(locMOmask > 0, "backprojected_moving_object_keypoints")){
                    namedWindow("Backprojected moving object keypoints", WINDOW_AUTOSIZE);
                    imshow("Backprojected moving object keypoints", locMOmask > 0);
                    waitKey(0);
                    destroyWindow("Backprojected moving object keypoints");
                }
            }

            Mat resMOmask;
            std::vector<vector<cv::Point>> hullPts(1);
            genMovObjHulls(locMOmask, keypointsMO, resMOmask, &hullPts[0]);

            /*namedWindow("Backprojected moving object area using convex hull", WINDOW_AUTOSIZE);
            imshow("Backprojected moving object area using convex hull", resMOmask > 0);*/

            locMOmask = (resMOmask(Rect(Point(posadd, posadd), imgSize)) > 0);
            if (keypointsMO.size() > 2) {
                Mat hullMat = Mat::zeros(imgSize, CV_8UC1);;
                drawContours(hullMat, hullPts, -1, Scalar(255), FILLED);
                locMOmask &= hullMat;
            }

            if (verbose & SHOW_BACKPROJECT_OCCLUSIONS_MOV_OBJ) {
                if(!writeIntermediateImg(locMOmask, "backprojected_moving_object_area_final")){
                    namedWindow("Backprojected moving object area final", WINDOW_AUTOSIZE);
                    imshow("Backprojected moving object area final", locMOmask);
                }
            }

            Mat overlaps = globMOmask & locMOmask;

            if (verbose & SHOW_BACKPROJECT_OCCLUSIONS_MOV_OBJ) {
                if(!writeIntermediateImg(overlaps, "overlap_with_other_moving_objects")){
                    namedWindow("Overlap with other moving objects", WINDOW_AUTOSIZE);
                    imshow("Overlap with other moving objects", overlaps);
                    waitKey(0);
                    destroyWindow("Overlap with other moving objects");
                    destroyWindow("Backprojected moving object area final");
                }
            }
//            destroyWindow("Backprojected moving object area using convex hull");


            if (cv::countNonZero(overlaps) > 0) {
                /*if(kpOutofR)
                {
                    throw SequenceException("Backprojected image coordinate of moving object is out of range.");
                }*/
                vector<int> filteredOccludedPts_idx_tmp;
                filteredOccludedPts_idx_tmp.reserve(actCorrsOnMovObjFromLast_IdxWorld.size());
                for (int j = 0; j < (int)keypointsMO.size(); ++j) {
                    if (overlaps.at<unsigned char>(keypointsMO[j]) == 0) {
                        movObj3DPtsCam[idx].push_back(cv::Point3d((double) (*filteredOccludedCamPts[idx])[j].x,
                                                                  (double) (*filteredOccludedCamPts[idx])[j].y,
                                                                  (double) (*filteredOccludedCamPts[idx])[j].z));
                        filteredOccludedPts_idx_tmp.push_back(actCorrsOnMovObjFromLast_IdxWorld[idx][j]);
                    }
                }
                if (movObj3DPtsCam[idx].empty()) {
                    delList.push_back(idx);
                }
                actCorrsOnMovObjFromLast_IdxWorld[idx] = filteredOccludedPts_idx_tmp;
            } else {
                for (int j = 0; j < (int)filteredOccludedCamPts[idx]->size(); ++j) {
                    movObj3DPtsCam[idx].push_back(cv::Point3d((double) (*filteredOccludedCamPts[idx])[j].x,
                                                              (double) (*filteredOccludedCamPts[idx])[j].y,
                                                              (double) (*filteredOccludedCamPts[idx])[j].z));
                }
            }
            globMOmask |= locMOmask;

            if (verbose & SHOW_BACKPROJECT_OCCLUSIONS_MOV_OBJ) {
                if(!writeIntermediateImg(globMOmask, "global_backprojected_moving_objects_mask")){
                    namedWindow("Global backprojected moving objects mask", WINDOW_AUTOSIZE);
                    imshow("Global backprojected moving objects mask", globMOmask);
                    waitKey(0);
                    destroyWindow("Global backprojected moving objects mask");
                }
            }
        }
    } else if (delList.empty()) {
        for (int j = 0; j < (int)filteredOccludedCamPts[0]->size(); ++j) {
            movObj3DPtsCam[0].push_back(cv::Point3d((double) (*filteredOccludedCamPts[0])[j].x,
                                                    (double) (*filteredOccludedCamPts[0])[j].y,
                                                    (double) (*filteredOccludedCamPts[0])[j].z));
        }
    }

    sort(delList.begin(), delList.end());
    if (!delList.empty()) {
        for (int i = (int) delList.size() - 1; i >= 0; i--) {
            movObj3DPtsCam.erase(movObj3DPtsCam.begin() + delList[i]);
            movObj3DPtsWorld.erase(movObj3DPtsWorld.begin() +
                                   delList[i]);//at this time, also moving objects that are only occluded are deleted
            movObjFrameEmerge.erase(movObjFrameEmerge.begin() + delList[i]);
            movObjImg12TP_InitIdxWorld.erase(movObjImg12TP_InitIdxWorld.begin() + delList[i]);
            movObjDepthClass.erase(movObjDepthClass.begin() + delList[i]);
            movObjWorldMovement.erase(movObjWorldMovement.begin() + delList[i]);
            actCorrsOnMovObjFromLast_IdxWorld.erase(actCorrsOnMovObjFromLast_IdxWorld.begin() + delList[i]);
        }
    }
}

void genStereoSequ::getCamPtsFromWorld() {
    if (staticWorld3DPts->empty()) {
        return;
    }

    actImgPointCloudFromLast.clear();
    actCorrsImg12TPFromLast_IdxWorld.clear();
//    pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_actImgPointCloudFromLast(staticWorld3DPts.makeShared());
    //Enables or disables filtering of occluded points for back-projecting existing 3D-world coorindinates to
    //the image plane as filtering occluded points is very time-consuming it can be disabled
    if(filter_occluded_points) {
        vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> camFilteredPtsParts;
        vector<vector<int>> camFilteredPtsParts_idx;
        const int partsx = 5;
        const int partsy = 5;
        const int partsxy = partsx * partsy;
//    bool success = getVisibleCamPointCloudSlices(staticWorld3DPts, camFilteredPtsParts, partsy, partsx, 0, 0);
        bool success = getVisibleCamPointCloudSlicesAndDepths(staticWorld3DPts,
                                                              camFilteredPtsParts,
                                                              camFilteredPtsParts_idx,
                                                              partsy,
                                                              partsx);
        if (!success) {
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr occludedPtsAll(new pcl::PointCloud<pcl::PointXYZ>());
        success = false;
        for (int i = 0; i < partsxy; i++) {
            if (camFilteredPtsParts[i].get() == nullptr) continue;
            if (camFilteredPtsParts[i]->empty()) continue;
            vector<int> filteredOccludedPts_idx;
            pcl::PointCloud<pcl::PointXYZ>::Ptr occludedPts(new pcl::PointCloud<pcl::PointXYZ>());
            bool success1 = filterNotVisiblePts(camFilteredPtsParts[i], filteredOccludedPts_idx, false, false,
                                                occludedPts);
            if (!success1) {
                filteredOccludedPts_idx.clear();
                occludedPts->clear();
                success |= filterNotVisiblePts(camFilteredPtsParts[i], filteredOccludedPts_idx, true, false,
                                               occludedPts);
                if (!filteredOccludedPts_idx.empty()) {
                    actCorrsImg12TPFromLast_IdxWorld.reserve(
                            actCorrsImg12TPFromLast_IdxWorld.size() + filteredOccludedPts_idx.size());
                    for (auto &j : filteredOccludedPts_idx) {
                        actCorrsImg12TPFromLast_IdxWorld.push_back(camFilteredPtsParts_idx[i][j]);
                    }
                }
                if (!occludedPts->empty()) {
                    occludedPtsAll->insert(occludedPtsAll->end(), occludedPts->begin(), occludedPts->end());
                }
            } else {
                success = true;
                actCorrsImg12TPFromLast_IdxWorld.reserve(
                        actCorrsImg12TPFromLast_IdxWorld.size() + filteredOccludedPts_idx.size());
                for (auto &j : filteredOccludedPts_idx) {
                    actCorrsImg12TPFromLast_IdxWorld.push_back(camFilteredPtsParts_idx[i][j]);
                }
                if (!occludedPts->empty()) {
                    occludedPtsAll->insert(occludedPtsAll->end(), occludedPts->begin(), occludedPts->end());
                }
            }
        }

        if (!actCorrsImg12TPFromLast_IdxWorld.empty() || !occludedPtsAll->empty()) {
            if (verbose & SHOW_BACKPROJECT_OCCLUSIONS_STAT_OBJ) {
                pcl::PointCloud<pcl::PointXYZ>::Ptr filteredOccludedPtsAll(new pcl::PointCloud<pcl::PointXYZ>());
                filteredOccludedPtsAll->reserve(actCorrsImg12TPFromLast_IdxWorld.size());
                for (auto &i : actCorrsImg12TPFromLast_IdxWorld) {
                    filteredOccludedPtsAll->push_back(staticWorld3DPts->at((size_t) i));
                }
                visualizeOcclusions(filteredOccludedPtsAll, occludedPtsAll, 1.0);
            }
        }

        if (!success) {
            if (actCorrsImg12TPFromLast_IdxWorld.empty()) {
                return;
            }
        }
    }else {
        bool success = getVisibleCamPointCloud(staticWorld3DPts, &actCorrsImg12TPFromLast_IdxWorld);
        if (!success) {
            return;
        }
    }

    for (auto& j : actCorrsImg12TPFromLast_IdxWorld) {
        cv::Point3d pt = Point3d((double) staticWorld3DPts->at((size_t)j).x, (double) staticWorld3DPts->at((size_t)j).y,
                                 (double) staticWorld3DPts->at((size_t)j).z);
        Mat ptm = Mat(pt, false).reshape(1, 3);
        ptm = absCamCoordinates[actFrameCnt].R.t() * (ptm -
                                                      absCamCoordinates[actFrameCnt].t);//physical memory of pt and ptm must be the same
        actImgPointCloudFromLast.emplace_back(pt);
    }
}

//Calculates the actual camera pose in camera coordinates in a different camera coordinate system (X forward, Y is up, and Z is right) to use the PCL filter FrustumCulling
void genStereoSequ::getActEigenCamPose() {
    Eigen::Affine3f cam_pose;
    cam_pose.setIdentity();
    Eigen::Vector3d te;
    Eigen::Matrix3d Re;
    cv::cv2eigen(absCamCoordinates[actFrameCnt].R, Re);
    cv::cv2eigen(absCamCoordinates[actFrameCnt].t, te);
    cam_pose.matrix().block<3, 3>(0, 0) = Re.cast<float>();
    cam_pose.matrix().block<3, 1>(0, 3) = te.cast<float>();

    Eigen::Matrix4f pose_orig = cam_pose.matrix();

    Eigen::Matrix4f cam2robot;
    cam2robot
            << 0, 0, 1.f, 0,//To convert from the traditional camera coordinate system (X right, Y down, Z forward) to (X is forward, Y is up, and Z is right)
            0, -1.f, 0, 0,
            1.f, 0, 0, 0,
            0, 0, 0, 1.f;
    //X_w = R_c2w * R_z^T * R_y^T * (R_y * R_z * X_c); cam2robot = R_z^T * R_y^T; X_w is in the normal (z forward, y down, x right) coordinate system
    //For the conversion from normal X_n to the other X_o (x forward, y up, z right) coordinate system: X_o = R_y * R_z * X_n (R_z rotation followed by R_y rotation)
    //R_y is a -90deg and R_z a 180deg rotation
    actCamPose = pose_orig * cam2robot;

    Eigen::Vector4d quat;
    MatToQuat(Re, quat);
    quatNormalise(quat);
    actCamRot = Eigen::Quaternionf(quat.cast<float>());
}

//Split the visible point cloud (through the camera) into a few slices to be able to use smaller leaf sizes in the
// pcl::VoxelGridOcclusionEstimation function as the number of useable voxels is bounded by a 32bit index
//Moreover split each slice in a near and a far part and use for each depth region a different voxel size based on the
//mean point cloud slice distance (bigger voxels for 3D points more distant to the camera)
bool genStereoSequ::getVisibleCamPointCloudSlicesAndDepths(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
                                                           std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cloudOut,
                                                           std::vector<std::vector<int>> &cloudOut_idx,
                                                           int fovDevideVertical,
                                                           int fovDevideHorizontal) {
    if (cloudIn->empty()) return false;

    if(!getVisibleCamPointCloudSlicesAndDepths(cloudIn, cloudOut_idx, fovDevideVertical, fovDevideHorizontal)){
        return false;
    }



    cloudOut.resize(cloudOut_idx.size());
    for(size_t i = 0; i < cloudOut_idx.size(); i++){
        if(!cloudOut_idx[i].empty()){
            cloudOut[i].reset(new pcl::PointCloud<pcl::PointXYZ>());
            cloudOut[i]->reserve(cloudOut_idx[i].size());
            for(auto& j : cloudOut_idx[i]){
                cloudOut[i]->push_back(cloudIn->at(j));
            }
        }
    }

    return true;
}

//Split the visible point cloud (through the camera) into a few slices to be able to use smaller leaf sizes in the
// pcl::VoxelGridOcclusionEstimation function as the number of useable voxels is bounded by a 32bit index
//Moreover split each slice in a near and a far part and use for each depth region a different voxel size based on the
//mean point cloud slice distance (bigger voxels for 3D points more distant to the camera)
bool genStereoSequ::getVisibleCamPointCloudSlicesAndDepths(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
                                                           std::vector<std::vector<int>> &cloudOut,
                                                           int fovDevideVertical,
                                                           int fovDevideHorizontal) {
    if (cloudIn->empty()) return false;

    //Get point cloud slices with 3D point depths in the camera coordinate system from near to mid
    int partsxy = fovDevideVertical * fovDevideHorizontal;
    std::vector<std::vector<int>> cloudsNear_idx;
    bool success1 = getVisibleCamPointCloudSlices(cloudIn,
                                                  cloudsNear_idx,
                                                  fovDevideVertical,
                                                  fovDevideHorizontal,
                                                  (float) actDepthNear,
                                                  (float) actDepthMid);

    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudsNear(cloudsNear_idx.size());
    for(size_t i = 0; i < cloudsNear_idx.size(); i++){
        cloudsNear[i].reset(new pcl::PointCloud<pcl::PointXYZ>());
        cloudsNear[i]->reserve(cloudsNear_idx[i].size());
        for(auto& j : cloudsNear_idx[i]){
            cloudsNear[i]->push_back(cloudIn->at((size_t)j));
        }
    }

    //Get point cloud slices with 3D point depths in the camera coordinate system from mid to far
    std::vector<std::vector<int>> cloudsFar_idx;
    bool success2 = getVisibleCamPointCloudSlices(cloudIn,
                                                  cloudsFar_idx,
                                                  fovDevideVertical,
                                                  fovDevideHorizontal,
                                                  (float) actDepthMid,
                                                  (float) (maxFarDistMultiplier * actDepthFar));

    //Check for duplicates at the borders
    if (success1 && success2) {
        vector<pair<size_t, size_t>> delList;
        for (size_t i = 0; i < cloudsNear_idx.size(); ++i) {
            for (size_t j = 0; j < cloudsNear_idx[i].size(); ++j) {
                for (size_t l = 0; l < cloudsFar_idx[i].size(); ++l) {
                    float dist = abs(cloudIn->at(cloudsNear_idx[i][j]).x - cloudIn->at(cloudsFar_idx[i][l]).x) +
                                 abs(cloudIn->at(cloudsNear_idx[i][j]).y - cloudIn->at(cloudsFar_idx[i][l]).y) +
                                 abs(cloudIn->at(cloudsNear_idx[i][j]).z - cloudIn->at(cloudsFar_idx[i][l]).z);
                    if (nearZero((double) dist)) {
                        delList.emplace_back(make_pair(i, l));
                    }
                }
            }
        }
        if (!delList.empty()) {
            for (int i = (int) delList.size() - 1; i >= 0; i--) {
                cloudsFar_idx[delList[i].first].erase(cloudsFar_idx[delList[i].first].begin() + delList[i].second);
            }
        }
    } else if (!success1 && !success2) {
        return false;
    }

    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudsFar(cloudsFar_idx.size());
    for(size_t i = 0; i < cloudsFar_idx.size(); i++){
        cloudsFar[i].reset(new pcl::PointCloud<pcl::PointXYZ>());
        cloudsFar[i]->reserve(cloudsFar_idx[i].size());
        for(auto& j : cloudsFar_idx[i]){
            cloudsFar[i]->push_back(cloudIn->at((size_t)j));
        }
    }

    //Filter near occluded 3D points with a smaller voxel size
    vector<vector<int>> filteredOccludedPtsNear((size_t)partsxy);
    if (success1) {
        success1 = false;
        for (int i = 0; i < partsxy; i++) {
            if (cloudsNear_idx[i].empty()) continue;
            bool success3 = filterNotVisiblePts(cloudsNear[i], filteredOccludedPtsNear[i], false, false);
            if (!success3) {
                vector<int> filteredOccludedPtsNear2;
                success3 = filterNotVisiblePts(cloudsNear[i], filteredOccludedPtsNear2, true, false);
                if (success3) {
                    filteredOccludedPtsNear[i] = filteredOccludedPtsNear2;
                    success1 = true;
                }
            } else {
                success1 = true;
            }
        }
    }

    //Filter far occluded 3D points with a bigger voxel size
    vector<vector<int>> filteredOccludedPtsFar((size_t)partsxy);
    if (success2) {
        success2 = false;
        for (int i = 0; i < partsxy; i++) {
            if (cloudsFar_idx[i].empty()) continue;
            bool success3 = filterNotVisiblePts(cloudsFar[i], filteredOccludedPtsFar[i], false, false);
            if (!success3) {
                vector<int> filteredOccludedPtsFar2;
                success3 = filterNotVisiblePts(cloudsFar[i], filteredOccludedPtsFar2, true, false);
                if (success3) {
                    filteredOccludedPtsFar[i] = filteredOccludedPtsFar2;
                    success2 = true;
                }
            } else {
                success2 = true;
            }
        }
    }

    //Combine near and far filtered point clouds
    success1 = false;
    cloudOut = std::vector<std::vector<int>>(partsxy);
    for(int i = 0; i < partsxy; i++){
        for(auto& j : filteredOccludedPtsNear[i]){
            cloudOut[i].push_back(cloudsNear_idx[i][j]);
        }
    }
    for (int i = 0; i < partsxy; i++) {
        if (!filteredOccludedPtsFar[i].empty()) {
            cloudOut[i].reserve(cloudOut[i].size() + filteredOccludedPtsFar[i].size());
            for(auto& j : filteredOccludedPtsFar[i]){
                cloudOut[i].push_back(cloudsFar_idx[i][j]);
            }
        }
        if (!cloudOut[i].empty()) success1 = true;
    }

    return success1;
}

bool genStereoSequ::getVisibleCamPointCloudSlices(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
                                                  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cloudOut,
                                                  int fovDevideVertical,
                                                  int fovDevideHorizontal,
                                                  float minDistance,
                                                  float maxDistance) {
    if (cloudIn->empty()) return false;

    pcl::PointCloud<pcl::PointXYZ>::Ptr camFilteredPts(new pcl::PointCloud<pcl::PointXYZ>());
    bool success = getVisibleCamPointCloud(cloudIn, camFilteredPts, 0, 0, 0, 0, minDistance, maxDistance);
    if (!success) {
        return false;
    }

    cloudOut = vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>((size_t)(fovDevideVertical * fovDevideHorizontal));
//    int checkSizePtCld = 0;
    success = false;
    for (int y = 0; y < fovDevideVertical; ++y) {
        for (int x = 0; x < fovDevideHorizontal; ++x) {
            cloudOut[y * fovDevideHorizontal + x].reset(new pcl::PointCloud<pcl::PointXYZ>());
            success |= getVisibleCamPointCloud(camFilteredPts,
                                               cloudOut[y * fovDevideHorizontal + x],
                                               fovDevideVertical,
                                               fovDevideHorizontal,
                                               y,
                                               x,
                                               minDistance,
                                               maxDistance);
//            checkSizePtCld += (int)cloudOut[y * fovDevideHorizontal + x]->size();
        }
    }
    //Remove duplicates
    vector<pair<size_t, size_t>> delList;
    for (size_t i = 0; i < cloudOut.size(); ++i) {
        for (size_t j = 0; j < cloudOut[i]->size(); ++j) {
            for (size_t k = i; k < cloudOut.size(); ++k) {
                for (size_t l = (k == i) ? (j + 1) : 0; l < cloudOut[k]->size(); ++l) {
                    float dist = abs(cloudOut[i]->at(j).x - cloudOut[k]->at(l).x) +
                                 abs(cloudOut[i]->at(j).y - cloudOut[k]->at(l).y) +
                                 abs(cloudOut[i]->at(j).z - cloudOut[k]->at(l).z);
                    if (nearZero((double) dist)) {
                        delList.emplace_back(make_pair(i, j));
                    }
                }
            }
        }
    }
    if (!delList.empty()) {
        for (int i = (int) delList.size() - 1; i >= 0; i--) {
            cloudOut[delList[i].first]->erase(cloudOut[delList[i].first]->begin() + delList[i].second);
//            checkSizePtCld--;
        }
    }
    //Typically, the returned number of 3D points within all slices is 1 smaller compared to the whole region (why?)
    /*if(checkSizePtCld != camFilteredPts->size()){
        cout << "Not working" << endl;
    }*/
    return success;
}

bool genStereoSequ::getVisibleCamPointCloudSlices(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
                                                  std::vector<std::vector<int>> &cloudOut,
                                                  int fovDevideVertical,
                                                  int fovDevideHorizontal,
                                                  float minDistance,
                                                  float maxDistance) {
    if (cloudIn->empty()) return false;

    std::vector<int> camFilteredPts_idx;
    bool success = getVisibleCamPointCloud(cloudIn, &camFilteredPts_idx, 0, 0, 0, 0, minDistance, maxDistance);
    if (!success) {
        return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr camFilteredPts(new pcl::PointCloud<pcl::PointXYZ>());
    camFilteredPts->reserve(camFilteredPts_idx.size());
    for(auto& i : camFilteredPts_idx){
        camFilteredPts->push_back(cloudIn->at((size_t)i));
    }

    cloudOut = vector<std::vector<int>>((size_t)(fovDevideVertical * fovDevideHorizontal));
//    int checkSizePtCld = 0;
    success = false;
    for (int y = 0; y < fovDevideVertical; ++y) {
        for (int x = 0; x < fovDevideHorizontal; ++x) {
            success |= getVisibleCamPointCloud(camFilteredPts,
                                               &cloudOut[y * fovDevideHorizontal + x],
                                               fovDevideVertical,
                                               fovDevideHorizontal,
                                               y,
                                               x,
                                               minDistance,
                                               maxDistance);
//            checkSizePtCld += (int)cloudOut[y * fovDevideHorizontal + x]->size();
        }
    }
    //Remove duplicates
    vector<pair<size_t, size_t>> delList;
    for (size_t i = 0; i < cloudOut.size(); ++i) {
        for (size_t j = 0; j < cloudOut[i].size(); ++j) {
            for (size_t k = i; k < cloudOut.size(); ++k) {
                for (size_t l = (k == i) ? (j + 1) : 0; l < cloudOut[k].size(); ++l) {
                    float dist = abs(camFilteredPts->at(cloudOut[i][j]).x - camFilteredPts->at(cloudOut[k][l]).x) +
                                 abs(camFilteredPts->at(cloudOut[i][j]).y - camFilteredPts->at(cloudOut[k][l]).y) +
                                 abs(camFilteredPts->at(cloudOut[i][j]).z - camFilteredPts->at(cloudOut[k][l]).z);
                    if (nearZero((double) dist)) {
                        delList.emplace_back(make_pair(i, j));
                    }
                }
            }
        }
    }
    if (!delList.empty()) {
        for (int i = (int) delList.size() - 1; i >= 0; i--) {
            cloudOut[delList[i].first].erase(cloudOut[delList[i].first].begin() + delList[i].second);
//            checkSizePtCld--;
        }
    }

    for(auto&& i : cloudOut){
        for(auto&& j : i){
            j = camFilteredPts_idx[j];
        }
    }
    //Typically, the returned number of 3D points within all slices is 1 smaller compared to the whole region (why?)
    /*if(checkSizePtCld != camFilteredPts->size()){
        cout << "Not working" << endl;
    }*/
    return success;
}

//Get part of a pointcloud visible in a camera.
// The function supports slicing of the truncated pyramid (frustum of pyramid) given the number of slices in vertical
// and horizontal direction and their 0-based indices (This function must be called for every slice seperately). Before
// using slicing, the function should be called without slicing and the results of this call should be provided to the
// function with slicing
//The output variables can only be of type pcl::PointCloud<pcl::PointXYZ>::Ptr or *std::vector<int>
template<typename T>
bool genStereoSequ::getVisibleCamPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
                                            T cloudOut,
                                            int fovDevideVertical,
                                            int fovDevideHorizontal,
                                            int returnDevPartNrVer,
                                            int returnDevPartNrHor,
                                            float minDistance,
                                            float maxDistance) {

    if (fovDevideVertical || fovDevideHorizontal || !nearZero((double) minDistance) ||
        !nearZero((double) maxDistance)) {
        CV_Assert((fovDevideVertical == 0) || ((returnDevPartNrVer >= 0) && (returnDevPartNrVer < fovDevideVertical)));
        CV_Assert((fovDevideHorizontal == 0) ||
                  ((returnDevPartNrHor >= 0) && (returnDevPartNrHor < fovDevideHorizontal)));
        CV_Assert(nearZero((double) minDistance) || ((minDistance >= (float) actDepthNear) &&
                                                     (((minDistance < maxDistance) || nearZero((double) maxDistance)) &&
                                                      (minDistance < (float) (maxFarDistMultiplier * actDepthFar)))));
        CV_Assert(nearZero((double) maxDistance) || ((maxDistance <= (float) (maxFarDistMultiplier * actDepthFar)) &&
                                                     ((minDistance < maxDistance) &&
                                                      (maxDistance > (float) actDepthNear))));
    }

    pcl::FrustumCulling<pcl::PointXYZ> fc;
    fc.setInputCloud(cloudIn);
    float verFOV = 2.f * std::atan((float) imgSize.height / (2.f * (float) K1.at<double>(1, 1)));
    float horFOV = 2.f * std::atan((float) imgSize.width / (2.f * (float) K1.at<double>(0, 0)));
    if (fovDevideVertical) {
        verFOV /= (float) fovDevideVertical;
    }
    if (fovDevideHorizontal) {
        horFOV /= (float) fovDevideHorizontal;
    }
    fc.setVerticalFOV(180.f * verFOV / (float) M_PI);
    fc.setHorizontalFOV(180.f * horFOV / (float) M_PI);

    float mimaDistance[2];
    if (nearZero((double) minDistance)) {
        mimaDistance[0] = (float) actDepthNear;
//        fc.setNearPlaneDistance((float) actDepthNear);
    } else {
        mimaDistance[0] = minDistance;
//        fc.setNearPlaneDistance(minDistance);
    }
    if (nearZero((double) maxDistance)) {
        mimaDistance[1] = (float) (maxFarDistMultiplier * actDepthFar);
//        fc.setFarPlaneDistance((float) (maxFarDistMultiplier * actDepthFar));
    } else {
        mimaDistance[1] = maxDistance;
//        fc.setFarPlaneDistance(maxDistance);
    }

    if (fovDevideVertical || fovDevideHorizontal) {
        Eigen::Matrix4f rotVer, rotHor, rotBoth;
        bool bothAngNotZero = false;
        double resalpha = 0;
        if (fovDevideVertical) {
            float angV = verFOV * ((float) returnDevPartNrVer - (float) fovDevideVertical / 2.f + 0.5f);
            if (!nearZero((double) angV)) {
                bothAngNotZero = true;
                resalpha = (double) angV;
            }
            /*rotVer
                    << 1.f, 0, 0, 0,
                    0, cos(angV), -sin(angV), 0,
                    0, sin(angV), cos(angV), 0,
                    0, 0, 0, 1.f;*/
            rotVer
                    << cos(angV), -sin(angV), 0, 0,
                    sin(angV), cos(angV), 0, 0,
                    0, 0, 1.f, 0,
                    0, 0, 0, 1.f;
            rotVer.transposeInPlace();
        } else {
            rotVer.setIdentity();
        }
        if (fovDevideHorizontal) {
            float angH = horFOV * ((float) returnDevPartNrHor - (float) fovDevideHorizontal / 2.f + 0.5f);
            if (nearZero((double) angH) && bothAngNotZero) {
                bothAngNotZero = false;
            } else if (!nearZero((double) angH)) {
                resalpha = (double) angH;
            }
            rotHor
                    << cos(angH), 0, sin(angH), 0,
                    0, 1.f, 0, 0,
                    -sin(angH), 0, cos(angH), 0,
                    0, 0, 0, 1.f;
            rotHor.transposeInPlace();
        } else {
            rotHor.setIdentity();
        }
        rotBoth = actCamPose * rotHor * rotVer;
        fc.setCameraPose(rotBoth);
        if (bothAngNotZero) {
            resalpha = rotDiff(actCamPose, rotBoth);
        }
        if (!nearZero(resalpha)) {
            resalpha = resalpha < 0 ? -resalpha : resalpha;
            while(resalpha > 2 * M_PI){
                resalpha -= 2 * M_PI;
            }
            if (!nearZero(resalpha)) {
                double tmp = cos(resalpha);
                tmp *= tmp * tmp;
                if(!nearZero(tmp)) {
                    //Adapt the minimum and maximum distance as the given distances which form 2 planes appear
                    // under a different angle compared to the original planes without slicing
                    mimaDistance[0] = (float)(((double)mimaDistance[0] * cos(2.0 * resalpha)) / tmp);
                    mimaDistance[1] = mimaDistance[1] / (float)tmp;
                }
            }
        }

    } else {
        fc.setCameraPose(actCamPose);
    }

    fc.setNearPlaneDistance(mimaDistance[0]);
    fc.setFarPlaneDistance(mimaDistance[1]);

    fc.filter(*cloudOut);

    if (cloudOut->empty())
        return false;

    return true;
}

//Filters occluded 3D points based on a voxel size corresponding to 1 pixel (when projected to the image plane) at near_depth + (medium depth - near_depth) / 2
//Returns false if more than 33% are occluded
bool genStereoSequ::filterNotVisiblePts(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut,
                                        bool useNearLeafSize,
                                        bool visRes,
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOccluded) {
    if (cloudIn->empty())
        return false;

    vector<int> cloudOut_idx;
    bool success = filterNotVisiblePts(cloudIn, cloudOut_idx, useNearLeafSize, visRes, cloudOccluded);

    if(!cloudOut_idx.empty()){
        cloudOut->reserve(cloudOut_idx.size());
        for(auto& i : cloudOut_idx){
            cloudOut->push_back(cloudIn->at((size_t)i));
        }
    }

    return success;
}

//Filters occluded 3D points based on a voxel size corresponding to 1 pixel (when projected to the image plane) at near_depth + (medium depth - near_depth) / 2
//Returns false if more than 33% are occluded
bool genStereoSequ::filterNotVisiblePts(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
                                        std::vector<int> &cloudOut,
                                        bool useNearLeafSize,
                                        bool visRes,
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOccluded) {
    if (cloudIn->empty())
        return false;

    cloudIn->sensor_origin_ = Eigen::Vector4f((float) absCamCoordinates[actFrameCnt].t.at<double>(0),
                                              (float) absCamCoordinates[actFrameCnt].t.at<double>(1),
                                              (float) absCamCoordinates[actFrameCnt].t.at<double>(2), 1.f);
    cloudIn->sensor_orientation_ = actCamRot;

    pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> voxelFilter;
    voxelFilter.setInputCloud(cloudIn);
    float leaf_size;
    pcl::PointXYZ cloudCentroid;
    getCloudCentroid(*cloudIn.get(), cloudCentroid);
    double usedZ;
    if (useNearLeafSize) {
        pcl::PointXYZ cloudDim;
        getCloudDimensionStdDev(*cloudIn.get(), cloudDim, cloudCentroid);
        double x[2], y[2], z[2];
        x[0] = (double) (cloudCentroid.x + cloudDim.x);
        x[1] = (double) (cloudCentroid.x - cloudDim.x);
        y[0] = (double) (cloudCentroid.y + cloudDim.y);
        y[1] = (double) (cloudCentroid.y - cloudDim.y);
        z[0] = (double) (cloudCentroid.z + cloudDim.z);
        z[1] = (double) (cloudCentroid.z - cloudDim.z);
        double minZ = DBL_MAX;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    Mat Xw = (Mat_<double>(3, 1) << x[i], y[j], z[k]);
                    Mat Xc = absCamCoordinates[actFrameCnt].R.t() * (Xw - absCamCoordinates[actFrameCnt].t);
                    double ptz = Xc.at<double>(2);
                    if ((ptz < minZ) && (ptz > actDepthNear)) {
                        minZ = ptz;
                    }
                }
            }
        }
        if (minZ > maxFarDistMultiplier * actDepthFar) {
            Mat Xw = (Mat_<double>(3, 1)
                    << (double) cloudCentroid.x, (double) cloudCentroid.y, (double) cloudCentroid.z);
            Mat Xc = absCamCoordinates[actFrameCnt].R.t() * (Xw - absCamCoordinates[actFrameCnt].t);
            minZ = Xc.at<double>(2);
        }
        usedZ = minZ;
        leaf_size = (float) minZ;
    } else {
        Mat Xw = (Mat_<double>(3, 1) << (double) cloudCentroid.x, (double) cloudCentroid.y, (double) cloudCentroid.z);
        Mat Xc = absCamCoordinates[actFrameCnt].R.t() * (Xw - absCamCoordinates[actFrameCnt].t);
        usedZ = Xc.at<double>(2);
        leaf_size = (float) usedZ;
    }
    if(nearZero(usedZ))
        usedZ = 1.0;
    leaf_size /= (float) K1.at<double>(0, 0);

    if(nearZero(leaf_size))
        leaf_size = 0.1;

    //Check if leaf size is too small for PCL (as there is a limitation within PCL)
    Eigen::Vector4f min_p, max_p;
    pcl::getMinMax3D(*cloudIn.get(), min_p, max_p);
    float d1, d2, d3;
    d1 = abs(max_p[0] - min_p[0]);
    d2 = abs(max_p[1] - min_p[1]);
    d3 = abs(max_p[2] - min_p[2]);
    int64_t dx = static_cast<int64_t>(d1 / leaf_size) + 1;
    int64_t dy = static_cast<int64_t>(d2 / leaf_size) + 1;
    int64_t dz = static_cast<int64_t>(d3 / leaf_size) + 1;
    auto maxIdxSize = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - 1;
    if ((dx * dy * dz) > maxIdxSize) {
        double kSi = (double) max(((csurr.rows - 1) / 2), 1);
        kSi = kSi > 3.0 ? 3.0 : kSi;
        leaf_size = (float) (kSi * usedZ / K1.at<double>(0, 0));
        if(nearZero(leaf_size))
            leaf_size = 0.1;
        dx = static_cast<int64_t>(d1 / leaf_size) + 1;
        dy = static_cast<int64_t>(d2 / leaf_size) + 1;
        dz = static_cast<int64_t>(d3 / leaf_size) + 1;
        while (((dx * dy * dz) > maxIdxSize) && ((kSi + DBL_EPSILON) < csurr.rows)) {
            kSi++;
            leaf_size = (float) (kSi * usedZ / K1.at<double>(0, 0));
            dx = static_cast<int64_t>(d1 / leaf_size) + 1;
            dy = static_cast<int64_t>(d2 / leaf_size) + 1;
            dz = static_cast<int64_t>(d3 / leaf_size) + 1;
        }
        if ((dx * dy * dz) > maxIdxSize) {
            double lNew = cbrt(ceil(100.0 * (double) d1 * (double) d2 * (double) d3 / (double) maxIdxSize) / 100.0);
            const double maxlsenlarge = 1.2;
            if (lNew > maxlsenlarge * (double) leaf_size) {
                //Go on without filtering
//                *cloudOut.get() = *cloudIn.get();
                cloudOut.resize(cloudIn->size());
                std::iota(cloudOut.begin(), cloudOut.end(), 0);
                return true;
            } else {
                leaf_size = (float)lNew;
                if(nearZero(leaf_size))
                    leaf_size = 0.1;
                dx = static_cast<int64_t>(d1 / leaf_size) + 1;
                dy = static_cast<int64_t>(d2 / leaf_size) + 1;
                dz = static_cast<int64_t>(d3 / leaf_size) + 1;
                double lsenlarge = 1.05;
                while (((dx * dy * dz) > maxIdxSize) && (lsenlarge < maxlsenlarge)) {
                    leaf_size *= lsenlarge;
                    dx = static_cast<int64_t>(d1 / leaf_size) + 1;
                    dy = static_cast<int64_t>(d2 / leaf_size) + 1;
                    dz = static_cast<int64_t>(d3 / leaf_size) + 1;
                    lsenlarge *= 1.05;
                }
                if ((dx * dy * dz) > maxIdxSize) {
                    //Go on without filtering
//                    *cloudOut.get() = *cloudIn.get();
                    cloudOut.resize(cloudIn->size());
                    std::iota(cloudOut.begin(), cloudOut.end(), 0);
                    return true;
                }
            }
        }
    }

    voxelFilter.setLeafSize(leaf_size, leaf_size,
                            leaf_size);//1 pixel (when projected to the image plane) at near_depth + (medium depth - near_depth) / 2
    try {
        voxelFilter.initializeVoxelGrid();
    }catch (exception &e){
        std::cerr << "Exception during filtering background: " << e.what() << endl;
        std::cerr << "Skipping filtering step." << endl;
        //Go on without filtering
//                    *cloudOut.get() = *cloudIn.get();
        cloudOut.resize(cloudIn->size());
        std::iota(cloudOut.begin(), cloudOut.end(), 0);
        return true;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOccluded_;//(new pcl::PointCloud<pcl::PointXYZ>);
    if (cloudOccluded.get() != nullptr) {
        cloudOccluded_ = cloudOccluded;
    } else {
        cloudOccluded_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }
    for (int i = 0; i < (int)cloudIn->size(); i++) {
        Eigen::Vector3i grid_coordinates = voxelFilter.getGridCoordinates(cloudIn->points[i].x,
                                                                          cloudIn->points[i].y,
                                                                          cloudIn->points[i].z);
        int grid_state;
        int ret = voxelFilter.occlusionEstimation(grid_state, grid_coordinates);
        if ((ret == 0) && (grid_state == 0)) {
            cloudOut.push_back(i);
        } else if ((ret == 0) && (verbose & SHOW_BACKPROJECT_OCCLUSIONS_MOV_OBJ)) {
            cloudOccluded_->push_back(cloudIn->points[i]);
        }
    }

    if (visRes && (verbose & SHOW_BACKPROJECT_OCCLUSIONS_MOV_OBJ)) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut_pts(new pcl::PointCloud<pcl::PointXYZ>);
        if(!cloudOut.empty()){
            cloudOut_pts->reserve(cloudOut.size());
            for(auto& i : cloudOut){
                cloudOut_pts->push_back(cloudIn->at((size_t)i));
            }
        }
        visualizeOcclusions(cloudOut_pts, cloudOccluded_, (double) leaf_size);
    }

    float fracOcc = (float) (cloudOut.size()) / (float) (cloudIn->size());
    if (fracOcc < 0.67)
        return false;

    return true;
}

void genStereoSequ::visualizeOcclusions(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVisible,
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOccluded,
                                        double ptSize) {
    if (cloudVisible->empty() && cloudOccluded->empty())
        return;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("Occlusions within an object"));

    Eigen::Affine3f m = initPCLViewerCoordinateSystems(viewer, absCamCoordinates[actFrameCnt].R,
                                                       absCamCoordinates[actFrameCnt].t);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr basic_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);


    for (auto &i : *cloudVisible.get()) {
        pcl::PointXYZRGB point;
        point.x = i.x;
        point.y = i.y;
        point.z = i.z;

        point.b = 0;
        point.g = 255;
        point.r = 0;
        basic_cloud_ptr->push_back(point);
    }
    for (auto &i : *cloudOccluded.get()) {
        pcl::PointXYZRGB point;
        point.x = i.x;
        point.y = i.y;
        point.z = i.z;

        point.b = 0;
        point.g = 0;
        point.r = 255;
        basic_cloud_ptr->push_back(point);
    }
    viewer->addPointCloud<pcl::PointXYZRGB>(basic_cloud_ptr, "visible and occluded points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ptSize,
                                             "visible and occluded points");

    setPCLViewerCamPars(viewer, m.matrix(), K1);

    startPCLViewer(viewer);
}

//Check if 2D correspondence is projection of 3D point
double genStereoSequ::project3DError(const cv::Mat &x, const cv::Mat &X, const cv::Mat &Ki){
    cv::Mat x1 = Ki * x;
    if(nearZero(x1.at<double>(2))) return DBL_MAX;
    x1 *= X.at<double>(2) / x1.at<double>(2);
    Mat diff = X - x1;
    return cv::norm(diff);
}

//Check, if 2D correspondences of both stereo cameras are projections of corresponding 3D point
bool genStereoSequ::checkCorrespondenceConsisty(const cv::Mat &x1, const cv::Mat &x2, const cv::Mat &X){
    double err1 = project3DError(x1, X, K1i);
    double err2 = project3DError(x2, actR * X + actT, K2i);
    double err = err1 + err2;
    bool test = nearZero(err / 10.0);
    return test;
}

//Set camera matrices from outside to be able to use function checkCorrespondenceConsisty
void genStereoSequ::setCamMats(const cv::Mat &K1_, const cv::Mat &K2_){
    K1 = K1_.clone();
    K2 = K2_.clone();
    K1i = K1.inv();
    K2i = K2.inv();
}

//Check, if 3D points are consistent with image projections
bool genStereoSequ::checkCorr3DConsistency(){
    CV_Assert((actCorrsImg1TPFromLast.cols == actCorrsImg2TPFromLast.cols) && ((int)actCorrsImg12TPFromLast_Idx.size() == actCorrsImg1TPFromLast.cols));
    for (int i = 0; i < actCorrsImg1TPFromLast.cols; ++i) {
        Mat x1 = actCorrsImg1TPFromLast.col(i);
        Mat x2 = actCorrsImg2TPFromLast.col(i);
        Mat X = Mat(actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[i]], true).reshape(1);
        const bool check1 = checkCorrespondenceConsisty(x1, x2, X);
        if (!check1){
            return false;
        }
    }
    return true;
}

//Perform the whole procedure of generating correspondences, new static, and dynamic 3D elements
void genStereoSequ::getNewCorrs() {
    updateFrameParameters();

    //Get pose of first camera in camera coordinates using a different coordinate system where X is forward, Y is up, and Z is right
    getActEigenCamPose();

    if (pars.nrMovObjs > 0) {
        cv::Mat movObjMask;
        int32_t corrsOnMovObjLF = 0;
        bool calcNewMovObj = true;
        if (actFrameCnt == 0) {
            movObjMask = Mat::zeros(imgSize, CV_8UC1);
        } else {
            // Update the 3D world coordinates of movObj3DPtsWorld based on direction and velocity
            updateMovObjPositions();

            //Calculate movObj3DPtsCam from movObj3DPtsWorld: Get 3D-points of moving objects that are visible in the camera and transform them from the world coordinate system into camera coordinate system
            getMovObjPtsCam();

            //Generate maps (masks) of moving objects by backprojection from 3D for the first and second stereo camera: movObjMaskFromLast, movObjMaskFromLast2; create convex hulls: convhullPtsObj; and
            //Check if some moving objects should be deleted
            backProjectMovObj();
            movObjMask = movObjMaskFromLast;
            corrsOnMovObjLF = actCorrsOnMovObjFromLast;

            //Generate seeds and areas for new moving objects
            calcNewMovObj = getNewMovObjs();
            if (calcNewMovObj) {
                calcNewMovObj = getSeedsAreasMovObj();
            }
        }
        if (calcNewMovObj) {
            std::vector<cv::Point_<int32_t>> seeds;
            std::vector<int32_t> areas;
            if (getSeedAreaListFromReg(seeds, areas)) {
                //Generate new moving objects and adapt the number of static correspondences per region
                generateMovObjLabels(movObjMask, seeds, areas, corrsOnMovObjLF, actStereoImgsOverlapMask);

                //Assign a depth category to each new moving object label and calculate all depth values for each label
                genNewDepthMovObj();

                //Generate correspondences and 3D points for new moving objects
                getMovObjCorrs();

                //Insert new 3D points (from moving objects) into world coordinate system
                transMovObjPtsToWorld();
            }
        } else {
            //Adapt the number of static correspondences based on the backprojected moving objects
            adaptNrStaticCorrsBasedOnMovCorrs(movObjMask);
            //Set the global mask for moving objects
            combMovObjLabelsAll = movObjMaskFromLast;
            movObjMask2All = movObjMaskFromLast2;
        }
    }

    //Get 3D points of static elements and store them to actImgPointCloudFromLast
    getCamPtsFromWorld();

    //Backproject static 3D points
    backProject3D();

    //Generate seeds for generating depth areas and include the seeds found by backprojection of the 3D points of the last frames
    checkDepthSeeds();

    //Generate depth areas for the current image and static elements
    genDepthMaps();

    //Generates correspondences and 3D points in the camera coordinate system (including false matches) from static scene elements
    getKeypoints();

    //Combine correspondences of static and moving objects
    combineCorrespondences();

    //Insert new 3D coordinates into the world coordinate system
    transPtsToWorld();

    //Combine indices to the 3D world coordinates of all correspondences (from last, new generated, moving objects)
    combineWorldCoordinateIndices();
}

//Start calculating the whole sequence
bool genStereoSequ::startCalc_internal() {
    static unsigned char init = 1;
    chrono::high_resolution_clock::time_point t1, t2;
    t1 = chrono::high_resolution_clock::now();
    if(init > 0) {
        actFrameCnt = 0;
        actCorrsPRIdx = 0;
        actStereoCIdx = 0;
        timePerFrame.clear();
        if(init == 2){
            resetInitVars();
        }
        init = 0;
    }

    if (actFrameCnt < totalNrFrames) {
        getNewCorrs();
        actFrameCnt++;
    }else{
        init = 2;
        return false;
    }

    t2 = chrono::high_resolution_clock::now();
    timePerFrame.emplace_back(chrono::duration_cast<chrono::microseconds>(t2 - t1).count());

    return true;
}

void genStereoSequ::startCalc(){
    if(!pars.parsAreValid){
        cout << "Provide parameters for generating a sequence!" << endl;
        return;
    }
    while(startCalc_internal());
}
