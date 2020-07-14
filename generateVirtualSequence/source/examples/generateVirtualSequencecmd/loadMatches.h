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

#ifndef GENERATEVIRTUALSEQUENCE_LOADMATCHES_H
#define GENERATEVIRTUALSEQUENCE_LOADMATCHES_H

#include <iostream>
#include <string>
//#include <stdio.h>
//#include <stdlib.h>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

struct sequMatches{
    /* Keypoints for the first (left or top) stereo cam (there is no 1:1 correspondence between
     * frameKeypoints1 and frameKeypoints2 as they are shuffled but the keypoint order of each
     * of them is the same compared to their corresponding descriptor Mat (rows))*/
    std::vector<cv::KeyPoint> frameKeypoints1;
    /* Keypoints for the second (right or bottom) stereo cam (there is no 1:1 correspondence between
     * frameKeypoints1 and frameKeypoints2 as they are shuffled but the keypoint order of each
     * of them is the same compared to their corresponding descriptor Mat (rows))*/
    std::vector<cv::KeyPoint> frameKeypoints2;
    /* Descriptors for first (left or top) stereo cam (there is no 1:1 correspondence between
     * frameDescriptors1 and frameDescriptors2 as they are shuffled but the descriptor order
     * is the same compared to its corresponding keypoint vector frameKeypoints1).
     * Descriptors corresponding to the same static 3D point (not for moving objects) in different
     * stereo frames are equal*/
    cv::Mat frameDescriptors1;
    /* Descriptors for second (right or bottom) stereo cam (there is no 1:1 correspondence between
     * frameDescriptors1 and frameDescriptors2 as they are shuffled but the descriptor order
     * the same compared to its corresponding keypoint vector frameKeypoints2).
     * Descriptors corresponding to the same static 3D point (not for moving objects) in different
     * stereo frames are similar*/
    cv::Mat frameDescriptors2;
    /* Matches between features of a single stereo frame. They are sorted based on the descriptor
     * distance (smallest first)*/
    std::vector<cv::DMatch> frameMatches;
    /* Indicates if a feature (frameKeypoints1 and corresponding frameDescriptors1) is an inlier.*/
    std::vector<bool> frameInliers;
    /* Keypoints in the second stereo image without a positioning error (in general, keypoints
     * in the first stereo image are without errors)*/
    std::vector<cv::KeyPoint> frameKeypoints2NoErr;
    /* Holds the homographies for all patches arround keypoints for warping the patch which is
     * then used to calculate the matching descriptor. Homographies corresponding to the same
     * static 3D point (not for moving objects) in different stereo frames are similar*/
    std::vector<cv::Mat> frameHomographies;
    /* Holds homographies for all patches arround keypoints in the first camera (for tracked features)
     * for warping the patch which is then used to calculate the matching descriptor.
     * Homographies corresponding to the same static 3D point in different stereo frames are similar
     */
    std::vector<cv::Mat> frameHomographiesCam1;
    /* Holds the keypoints from the images used to extract patches (image indices for keypoints
     * are stored in srcImgPatchKpImgIdx1)*/
    std::vector<cv::KeyPoint> srcImgPatchKp1;
    /* Holds the image indices of the images used to extract patches for every keypoint in srcImgPatchKp1 (same order) */
    std::vector<size_t> srcImgPatchKpImgIdx1;
    /* Holds the keypoints from the images used to extract patches for the second keypoint of a match.
     * (image indices for keypoints are stored in srcImgPatchKpImgIdx2) */
    std::vector<cv::KeyPoint> srcImgPatchKp2;
    /* Holds the image indices of the images used to extract patches for every keypoint in srcImgPatchKp2 (same order) */
    std::vector<size_t> srcImgPatchKpImgIdx2;
    /* Specifies the type of a correspondence (TN from static (=4) or TN from moving (=5) object,
     * or TP from a new static (=0), a new moving (=1), an old static (=2), or an old moving (=3)
     * object (old means, that the corresponding 3D point emerged before this stereo frame and
     * also has one or more correspondences in a different stereo frame))*/
    std::vector<int> corrType;
    /* Indices for final correspondences to point from reordered frameKeypoints1, frameKeypoints2,
     * frameDescriptors1, frameDescriptors2, frameHomographies, ... to corresponding 3D information
     * like combCorrsImg1TP, combCorrsImg2TP, combCorrsImg12TP_IdxWorld2, ... */
    std::vector<size_t> idxs_match23D1, idxs_match23D2;
};

//Order of correspondences in combined Mat combCorrsImg1TP, combCorrsImg2TP, and comb3DPts
struct CorrOrderTP{
    CorrOrderTP()
    {
        statTPfromLast = 0;
        statTPnew = 1;
        movTPfromLast = 2;
        movTPnew = 3;
    }
    unsigned char statTPfromLast;
    unsigned char statTPnew;
    unsigned char movTPfromLast;
    unsigned char movTPnew;
};

struct data3D{
    /* Frame number*/
    size_t actFrameCnt = 0;
    /* Actual rotation matrix of the stereo rig: x2 = actR * x1 + actT*/
    cv::Mat actR;
    /* Actual translation vector of the stereo rig: x2 = actR * x1 + actT*/
    cv::Mat actT;
    /* Actual correct camera matrix of camera 1*/
    cv::Mat K1;
    /* Actual correct camera matrix of camera 2*/
    cv::Mat K2;
    /* Actual distorted camera matrix of camera 1*/
    cv::Mat actKd1;
    /* Actual distorted camera matrix of camera 2*/
    cv::Mat actKd2;
    /* Lower border of near depths for the actual camera configuration */
    double actDepthNear = 0;
    /* Upper border of near and lower border of mid depths for the actual camera configuration */
    double actDepthMid = 0;
    /* Upper border of mid and lower border of far depths for the actual camera configuration */
    double actDepthFar = 0;
    /* Combined TP correspondences (static and moving objects). Size: 3xn; Last row should be 1.0; Both Mat must have the same size. */
    cv::Mat combCorrsImg1TP, combCorrsImg2TP;
    /* Combined 3D points corresponding to matches combCorrsImg1TP and combCorrsImg2TP */
    std::vector<cv::Point3d> comb3DPts;
    /* Index to the corresponding world 3D point within staticWorld3DPts and movObj3DPtsWorld of combined
     * TP correspondences (static and moving objects) in combCorrsImg1TP and combCorrsImg2TP.
     * Indices on static objects are positive. Indices on moving objects are negative: The first 32bit hold
     * the vector index for movObj3DPtsWorld plus 1 and the next 31bit hold the 3D world coordinate
     * index of the corresponding within the moving object: idx = -1 * ((nr_mov_obj + 1) | (index_coordinate << 32)) */
    std::vector<int64_t> combCorrsImg12TP_IdxWorld;
    /* Index to the corresponding world 3D point within movObj3DPtsWorldAllFrames of TP of moving objects in
     * combCorrsImg1TP and combCorrsImg2TP (static correspondences are positive like in combCorrsImg12TP_IdxWorld).
     * Indices on moving objects are negative: The first 32bit hold the last vector index for
     * movObj3DPtsWorldAllFrames plus 1 and the next 24bit hold the frame number when the moving object
     * emerged (The sum of all moving objects before this frame number (use nrMovingObjPerFrame to calculate it) +
     * the moving object number within this frame number which is also included in this index lead to the
     * correct vector index for movObj3DPtsWorldAllFrames). The last 8bit hold the moving object
     * number (index + 1): idx = -1 * ((index_coordinate << 32) | (frame_number << 8) | (nr_mov_obj + 1)) */
    std::vector<int64_t> combCorrsImg12TP_IdxWorld2;
    /* Holds the number of visible moving objects per frame. Their sum corresponds to the number of elements in movObj3DPtsWorldAllFrames */
    std::vector<unsigned int> nrMovingObjPerFrame;
    /* Holds the frame count when the moving object emerged, its initial position in vector
     * combCorrsImg12TPContMovObj_IdxWorld, and its corresponding movObjWorldMovement: e.g. (actFrameCnt, pos, movement_vector) */
    std::vector<std::tuple<size_t, size_t, cv::Mat>> movObjFrameEmerge;
    /* Similar to combCorrsImg12TP_IdxWorld but the vector indices for moving objects do NOT correspond with
     * vector elements in movObj3DPtsWorld but with a consecutive number pointing to moving object pointclouds
     * that were saved after they emerged. The index number in the first 8 bits can also be found in the
     * corresponding file name where the PCL pointcloud was saved to. */
    std::vector<int64_t> combCorrsImg12TPContMovObj_IdxWorld;
    /* Combined TN correspondences (static and moving objects). Size: 3xn; Last row should be 1.0; Both Mat must have the same size. */
    cv::Mat combCorrsImg1TN, combCorrsImg2TN;
    /* Number of overall TP and TN correspondences (static and moving objects) */
    int combNrCorrsTP = 0, combNrCorrsTN = 0;
    /* Distance values of all (static and moving objects) TN keypoint locations in the 2nd image to the location
     * that would be a perfect correspondence to the TN in image 1. If the value is >= 50, the "perfect location" would be outside the image */
    std::vector<double> combDistTNtoReal;
    /* Final number of new generated TP correspondences for static objects. Corresponds to the number of columns in actCorrsImg1TP */
    int finalNrTPStatCorrs = 0;
    /* Final number of new generated TP correspondences for moving objects. Corresponds to the sum of number of columns in movObjCorrsImg1TP */
    int finalNrTPMovCorrs = 0;
    /* Final number of backprojected TP correspondences for static objects. Corresponds to the number of columns in actCorrsImg1TPFromLast */
    int finalNrTPStatCorrsFromLast = 0;
    /* Final number of backprojected TP correspondences for moving objects. Corresponds to the sum of number of columns in movObjCorrsImg1TPFromLast */
    int finalNrTPMovCorrsFromLast = 0;
    /* Final number of TN correspondences for static objects. Corresponds to the number of columns in actCorrsImg1TN */
    int finalNrTNStatCorrs = 0;
    /* Final number of TN correspondences for moving objects. Corresponds to the sum of number
     * of columns in movObjCorrsImg1TNFromLast and movObjCorrsImg1TN */
    int finalNrTNMovCorrs = 0;
    /* Order of correspondences in combined Mat combCorrsImg1TP, combCorrsImg2TP, and comb3DPts */
    CorrOrderTP combCorrsImg12TPorder = CorrOrderTP();
    /* Indicates that TN correspondences of static objects are located at the beginning of Mats combCorrsImg1TN and combCorrsImg2TN */
    bool combCorrsImg12TNstatFirst = true;
};

struct FrameToFrameMatches{
    /* Keypoints in first stereo camera of frame i-1 */
    std::vector<cv::KeyPoint> frameKeypoints_1_1;
    /* Keypoints in first stereo camera of frame i */
    std::vector<cv::KeyPoint> frameKeypoints_2_1;
    /* Keypoints in second stereo camera of frame i-1 */
    std::vector<cv::KeyPoint> frameKeypoints_1_2;
    /* Keypoints in second stereo camera of frame i */
    std::vector<cv::KeyPoint> frameKeypoints_2_2;
    /* Descriptors in first stereo camera of frame i-1 */
    cv::Mat descriptors_1_1;
    /* Descriptors in first stereo camera of frame i */
    cv::Mat descriptors_2_1;
    /* Descriptors in first stereo camera of frame i */
    cv::Mat descriptors_1_2;
    /* Descriptors in first stereo camera of frame i */
    cv::Mat descriptors_2_2;
    /* Matches in first stereo camera of frame i-1 and i */
    std::vector<cv::DMatch> matches_1;
    /* Matches in second stereo camera of frame i-1 and i */
    std::vector<cv::DMatch> matches_2;
    /* Frame number i-1 */
    size_t frameNr = 0;
    /* Relative rotation matrix from first stereo camera of frame i-1 to first stereo camera of frame i */
    cv::Mat R_rel_1;
    /* Relative rotation matrix from second stereo camera of frame i-1 to second stereo camera of frame i */
    cv::Mat R_rel_2;
    /* Relative translation vector from first stereo camera of frame i-1 to first stereo camera of frame i */
    cv::Mat t_rel_1;
    /* Relative translation vector from second stereo camera of frame i-1 to second stereo camera of frame i */
    cv::Mat t_rel_2;
};

struct matchSequParameters {
    /* Path containing the images for producing keypoint patches */
    std::string imgPath;
    /* image pre- and/or postfix (supports wildcards & subfolders) for images within imgPath */
    std::string imgPrePostFix;
    /* Name of keypoint detector */
    std::string keyPointType;
    /* Name of descriptor extractor */
    std::string descriptorType;
    /* Keypoint detector error (true) or error normal distribution (false) */
    bool keypPosErrType = false;
    /* Keypoint error distribution (mean, std) for the matching keypoint location */
    std::pair<double, double> keypErrDistr = std::make_pair(0, 0);
    /* Noise (mean, std) on the image intensity for descriptor calculation */
    std::pair<double, double> imgIntNoise = std::make_pair(0, 0);
    /* Minimal and maximal percentage (0 to 1.0) of repeated patterns (image patches) between stereo cameras. */
    std::pair<double, double> repeatPatternPortStereo = std::make_pair(0, 0);
    /* Minimal and maximal percentage (0 to 1.0) of repeated patterns (image patches) from frame to frame. */
    std::pair<double, double> repeatPatternPortFToF = std::make_pair(0, 0);
    /* If true, tracked image patch in the first stereo image are distorted */
    bool distortPatchCam1 = false;
    /* Portion of GT matches from Oxford dataset (GT = homographies) */
    double oxfordGTMportion = 0;
    /* Portion of GT matches from KITTI dataset (GT = flow, disparity) */
    double kittiGTMportion = 0;
    /* Portion of GT matches from MegaDepth dataset (GT = depth) */
    double megadepthGTMportion = 0;
    /* Portion of GT matches (GTM) compared to warped patch correspondences if multiple datasets are used as source (Oxford, KITTI, MegaDepth) */
    double GTMportion = 0;
    /* Portion of TN that should be drawn from warped image patches (and not from GTM). */
    double WarpedPortionTN = 0;
    /* Portion of TN that should be from GTM or from different image patches (first <-> second stereo camera). */
    double portionGrossTN = 0;
    /* Folder name including matches */
    string hashMatchingPars;
};

struct Poses
{
    Poses()
    {
        R = cv::Mat::eye(3, 3, CV_64FC1);
        t = cv::Mat::zeros(3, 1, CV_64FC1);
    }
    Poses(const cv::Mat &R_, const cv::Mat &t_) : R(R_), t(t_) {}

    cv::Mat R;
    cv::Mat t;
};

struct depthPortion{
    depthPortion(): near(0), mid(0), far(0){}

    depthPortion(double near_, double mid_, double far_): near(near_), mid(mid_), far(far_){}

    double near;
    double mid;
    double far;
};

enum depthClass{
    NEAR = 0x01,
    MID = 0x02,
    FAR = 0x04
};

struct sequParameters{
    /* Number of different stereo camera configurations */
    size_t nrStereoConfs = 0;
    /* Inlier ratio for every frame */
    std::vector<double> inlRat;
    /* Total number of frames */
    size_t totalNrFrames = 0;
    /* Absolute number of correspondences (TP+TN) per frame */
    std::vector<size_t> nrCorrs;
    /* Absolute coordinates of the camera centres (left or bottom cam of stereo rig) for every frame;
     * Includes the rotation from the camera into world and the position of the camera centre C in the
     * world: X_world  = R * X_cam + t (t corresponds to C in this case); X_cam = R^T * X_world - R^T * t */
    std::vector<Poses> absCamCoordinates;
    /* Different relative stereo rotation matrices */
    std::vector<cv::Mat> R_stereo;
    /* Different relative stereo translation vectors */
    std::vector<cv::Mat> t_stereo;
    /* Sum over the number of moving objects in every frame */
    size_t nrMovObjAllFrames = 0;
    /* Camera matrix of first stereo cameras */
    cv::Mat K1;
    /* Camera matrix of second stereo cameras */
    cv::Mat K2;
    /* Size of the images */
    cv::Size imgSize;
    /* # of Frames per camera configuration */
    size_t nFramesPerCamConf = 0;
    /* Inlier ratio range */
    std::pair<double, double> inlRatRange = std::make_pair(0, 0);
    /* Inlier ratio change rate from pair to pair. If 0, the inlier ratio within the given range is always
     * the same for every image pair. If 100, the inlier ratio is chosen completely random within the given range.
     * For values between 0 and 100, the inlier ratio selected is not allowed to change more than this factor from the last inlier ratio. */
    double inlRatChanges = 0;
    /* # true positives range */
    std::pair<size_t, size_t> truePosRange = std::make_pair(0, 0);
    /* True positives change rate from pair to pair. If 0, the true positives within the given range are
     * always the same for every image pair. If 100, the true positives are chosen completely random within the given range.
     * For values between 0 and 100, the true positives selected are not allowed to change more than this factor from the true positives. */
    double truePosChanges = 0;
    /* min. distance between keypoints */
    double minKeypDist = 0;
    /* portion of correspondences at depths */
    depthPortion corrsPerDepth;
    /* List of portions of image correspondences at regions (Matrix must be 3x3). Maybe doesnt hold: Also depends on 3D-points from prior frames. */
    std::vector<cv::Mat> corrsPerRegion;
    /* Repeat rate of portion of correspondences at regions. If more than one matrix of portions of
     * correspondences at regions is provided, this number specifies the number of frames for which such a
     * matrix is valid. After all matrices are used, the first one is used again. If 0 and no matrix of portions of
     * correspondences at regions is provided, as many random matrizes as frames are randomly generated. */
    size_t corrsPerRegRepRate = 0;
    /* Portion of depths per region (must be 3x3). For each of the 3x3=9 image regions, the portion of near, mid, and far depths
     * can be specified. If the overall depth definition is not met, this tensor is adapted.
     * Maybe doesnt hold: Also depends on 3D - points from prior frames. */
    std::vector<std::vector<depthPortion>> depthsPerRegion;
    /* Min and Max number of connected depth areas per region (must be 3x3). The minimum number (first) must be larger 0.
     * The maximum number is bounded by the minimum area which is 16 pixels. Maybe doesnt hold: Also depends on 3D - points from prior frames. */
    std::vector<std::vector<std::pair<size_t, size_t>>> nrDepthAreasPReg;

    //Paramters for camera and object movements

    /* Movement direction or track of the cameras (Mat must be 3x1). If 1 vector: Direction in the form [tx, ty, tz].
     * If more vectors: absolute position edges on a track.  The scaling of the track is calculated using the velocity
     * information(The last frame is located at the last edge); tz is the main viewing direction of the first camera
     * which can be changed using the rotation vector for the camera centre.The camera rotation during movement is
     * based on the relative movement direction(like a fixed stereo rig mounted on a car). */
    std::vector<cv::Mat> camTrack;
    /* Relative velocity of the camera movement (value between 0 and 10; must be larger 0).
     * The velocity is relative to the baseline length between the stereo cameras */
    double relCamVelocity;
    /* Rotation matrix of the first camera centre. This rotation can change the camera orientation for which
     * without rotation the z - component of the relative movement vector coincides with the principal axis of the camera.
     * Rotation matrix must be generated using the form R_y * R_z * R_x. */
    cv::Mat R;
    /* Number of moving objects in the scene */
    size_t nrMovObjs = 0;
    /* Possible starting positions of moving objects in the image (must be 3x3 boolean (CV_8UC1)) */
    cv::Mat startPosMovObjs;
    /* Relative area range of moving objects. Area range relative to the image area at the beginning. */
    std::pair<double, double> relAreaRangeMovObjs = std::make_pair(0, 0);
    /* Depth of moving objects. Moving objects are always visible and not covered by other static objects.
     * If the number of paramters is 1, this depth is used for every object. If the number of paramters is
     * equal "nrMovObjs", the corresponding depth is used for every object. If the number of parameters is
     * smaller and between 2 and 3, the depths for the moving objects are selected uniformly distributed from
     * the given depths. For a number of paramters larger 3 and unequal to "nrMovObjs", a portion for every depth
     * that should be used can be defined (e.g. 3 x far, 2 x near, 1 x mid -> 3 / 6 x far, 2 / 6 x near, 1 / 6 x mid). */
    std::vector<depthClass> movObjDepth;
    /* Movement direction of moving objects relative to camera movementm (must be 3x1). The movement direction is
     * linear and does not change if the movement direction of the camera changes.The moving object is removed,
     * if it is no longer visible in both stereo cameras. */
    cv::Mat movObjDir;
    /* Relative velocity range of moving objects based on relative camera velocity. Values between 0 and 100; Must be larger 0; */
    std::pair<double, double> relMovObjVelRange = std::make_pair(0, 0);
    /* Minimal portion of correspondences on moving objects for removing them. If the portion of visible
     * correspondences drops below this value, the whole moving object is removed. Zero means, that the moving object
     * is only removed if there is no visible correspondence in the stereo pair. One means, that a single
     * missing correspondence leads to deletion. Values between 0 and 1; */
    double minMovObjCorrPortion = 0;
    /* Portion of correspondences on moving object (compared to static objects). It is limited by the
     * size of the objects visible in the images and the minimal distance between correspondences. */
    double CorrMovObjPort = 0;
    /* Minimum number of moving objects over the whole track. If the number of moving obects drops below this
     * number during camera movement, as many new moving objects are inserted until "nrMovObjs" is reached.
     * If 0, no new moving objects are inserted if every preceding object is out of sight. */
    size_t minNrMovObjs = 0;
    /* Minimal and maximal percentage (0 to 1.0) of random distortion of the camera matrices K1 & K2
     * based on their initial values (only the focal lengths and image centers are randomly distorted) */
    std::pair<double, double> distortCamMat = std::make_pair(0, 0);
};

FileStorage& operator << (FileStorage& fs, bool &value)
{
    if(value){
        return (fs << 1);
    }

    return (fs << 0);
}

void operator >> (const FileNode& n, bool& value)
{
    int bVal;
    n >> bVal;
    value = bVal != 0;
}

FileStorage& operator << (FileStorage& fs, int64_t &value)
{
    string strVal = std::to_string(value);
    return (fs << strVal);
}

void operator >> (const FileNode& n, int64_t& value)
{
    string strVal;
    n >> strVal;
    value = std::stoll(strVal);
}

FileNodeIterator& operator >> (FileNodeIterator& it, int64_t & value)
{
    *it >> value;
    return ++it;
}

bool readMatchesFromDisk(const std::string &filename,
                         sequMatches &sm){
    FileStorage fs = FileStorage(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    fs["frameKeypoints1"] >> sm.frameKeypoints1;
    fs["frameKeypoints2"] >> sm.frameKeypoints2;
    fs["frameDescriptors1"] >> sm.frameDescriptors1;
    fs["frameDescriptors2"] >> sm.frameDescriptors2;
    fs["frameMatches"] >> sm.frameMatches;

    FileNode n = fs["frameInliers"];
    if (n.type() != FileNode::SEQ) {
        cerr << "frameInliers is not a sequence! FAIL" << endl;
        return false;
    }
    sm.frameInliers.clear();
    FileNodeIterator it = n.begin(), it_end = n.end();
    while ( it != it_end) {
        bool inli = false;
        it >> inli;
        sm.frameInliers.push_back(inli);
    }

    fs["frameKeypoints2NoErr"] >> sm.frameKeypoints2NoErr;

    n = fs["frameHomographies"];
    if (n.type() != FileNode::SEQ) {
        cerr << "frameHomographies is not a sequence! FAIL" << endl;
        return false;
    }
    sm.frameHomographies.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        Mat m;
        it >> m;
        sm.frameHomographies.emplace_back(m.clone());
    }

    n = fs["frameHomographiesCam1"];
    if (n.type() != FileNode::SEQ) {
        cerr << "frameHomographiesCam1 is not a sequence! FAIL" << endl;
        return false;
    }
    sm.frameHomographiesCam1.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        Mat m;
        it >> m;
        sm.frameHomographiesCam1.emplace_back(m.clone());
    }

    fs["srcImgPatchKp1"] >> sm.srcImgPatchKp1;

    n = fs["srcImgPatchKpImgIdx1"];
    if (n.type() != FileNode::SEQ) {
        cerr << "srcImgPatchKpImgIdx1 is not a sequence! FAIL" << endl;
        return false;
    }
    sm.srcImgPatchKpImgIdx1.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int tmp = 0;
        it >> tmp;
        sm.srcImgPatchKpImgIdx1.push_back(static_cast<size_t>(tmp));
    }

    fs["srcImgPatchKp2"] >> sm.srcImgPatchKp2;

    n = fs["srcImgPatchKpImgIdx2"];
    if (n.type() != FileNode::SEQ) {
        cerr << "srcImgPatchKpImgIdx2 is not a sequence! FAIL" << endl;
        return false;
    }
    sm.srcImgPatchKpImgIdx2.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int tmp = 0;
        it >> tmp;
        sm.srcImgPatchKpImgIdx2.push_back(static_cast<size_t>(tmp));
    }

    n = fs["corrType"];
    if (n.type() != FileNode::SEQ) {
        cerr << "corrType is not a sequence! FAIL" << endl;
        return false;
    }
    sm.corrType.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int tmp = 0;
        it >> tmp;
        sm.corrType.push_back(tmp);
    }

    n = fs["idxs_match23D1"];
    if (n.type() != FileNode::SEQ) {
        cerr << "idxs_match23D1 is not a sequence! FAIL" << endl;
        return false;
    }
    sm.idxs_match23D1.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int tmp = 0;
        it >> tmp;
        sm.idxs_match23D1.push_back(static_cast<size_t>(tmp));
    }
    n = fs["idxs_match23D2"];
    if (n.type() != FileNode::SEQ) {
        cerr << "idxs_match23D2 is not a sequence! FAIL" << endl;
        return false;
    }
    sm.idxs_match23D2.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int tmp = 0;
        it >> tmp;
        sm.idxs_match23D2.push_back(static_cast<size_t>(tmp));
    }
    fs.release();

    return true;
}

bool readCamParsFromDisk(const std::string &filename,
                         data3D &sm){
    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    int tmp = 0;

    fs["actFrameCnt"] >> tmp;
    sm.actFrameCnt = (size_t) tmp;
    fs["actR"] >> sm.actR;
    fs["actT"] >> sm.actT;
    fs["K1"] >> sm.K1;
    fs["K2"] >> sm.K2;
    fs["actKd1"] >> sm.actKd1;
    fs["actKd2"] >> sm.actKd2;

    fs["actDepthNear"] >> sm.actDepthNear;
    fs["actDepthMid"] >> sm.actDepthMid;
    fs["actDepthFar"] >> sm.actDepthFar;

    fs["combCorrsImg1TP"] >> sm.combCorrsImg1TP;
    fs["combCorrsImg2TP"] >> sm.combCorrsImg2TP;

    FileNode n = fs["comb3DPts"];
    if (n.type() != FileNode::SEQ) {
        cerr << "comb3DPts is not a sequence! FAIL" << endl;
        return false;
    }
    sm.comb3DPts.clear();
    FileNodeIterator it = n.begin(), it_end = n.end();
    while (it != it_end) {
        cv::Point3d pt;
        it >> pt;
        sm.comb3DPts.push_back(pt);
    }

    n = fs["combCorrsImg12TP_IdxWorld"];
    if (n.type() != FileNode::SEQ) {
        cerr << "combCorrsImg12TP_IdxWorld is not a sequence! FAIL" << endl;
        return false;
    }
    sm.combCorrsImg12TP_IdxWorld.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int64_t val;
        it >> val;
        sm.combCorrsImg12TP_IdxWorld.push_back(val);
    }

    n = fs["combCorrsImg12TP_IdxWorld2"];
    if(!n.empty()){
        if (n.type() != FileNode::SEQ) {
            cerr << "combCorrsImg12TP_IdxWorld2 is not a sequence! FAIL" << endl;
            return false;
        }
        sm.combCorrsImg12TP_IdxWorld2.clear();
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            int64_t val;
            it >> val;
            sm.combCorrsImg12TP_IdxWorld2.push_back(val);
        }
    }

    n = fs["nrMovingObjPerFrame"];
    if(!n.empty()){
        if (n.type() != FileNode::SEQ) {
            cerr << "nrMovingObjPerFrame is not a sequence! FAIL" << endl;
            return false;
        }
        sm.nrMovingObjPerFrame.clear();
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            int val;
            it >> val;
            sm.nrMovingObjPerFrame.push_back(static_cast<unsigned int>(val));
        }
    }

    n = fs["movObjFrameEmerge"];
    if(!n.empty()){
        if (n.type() != FileNode::SEQ) {
            cerr << "movObjFrameEmerge is not a sequence! FAIL" << endl;
            return false;
        }
        sm.movObjFrameEmerge.clear();
        it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it) {
            FileNode n1 = *it;
            Mat move_vect;
            int fc, pos;
            n1["fc"] >> fc;
            n1["pos"] >> pos;
            n1["move_vect"] >> move_vect;
            sm.movObjFrameEmerge.emplace_back(make_tuple(static_cast<size_t>(fc), static_cast<size_t>(pos), move_vect.clone()));
        }
    }

    n = fs["combCorrsImg12TPContMovObj_IdxWorld"];
    if (n.type() != FileNode::SEQ) {
        cerr << "combCorrsImg12TPContMovObj_IdxWorld is not a sequence! FAIL" << endl;
        return false;
    }
    sm.combCorrsImg12TPContMovObj_IdxWorld.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int64_t val;
        it >> val;
        sm.combCorrsImg12TPContMovObj_IdxWorld.push_back(val);
    }

    fs["combCorrsImg1TN"] >> sm.combCorrsImg1TN;
    fs["combCorrsImg2TN"] >> sm.combCorrsImg2TN;

    fs["combNrCorrsTP"] >> sm.combNrCorrsTP;
    fs["combNrCorrsTN"] >> sm.combNrCorrsTN;

    n = fs["combDistTNtoReal"];
    if (n.type() != FileNode::SEQ) {
        cerr << "combDistTNtoReal is not a sequence! FAIL" << endl;
        return false;
    }
    sm.combDistTNtoReal.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        double dist = 0;
        it >> dist;
        sm.combDistTNtoReal.push_back(dist);
    }

    fs["finalNrTPStatCorrs"] >> sm.finalNrTPStatCorrs;

    fs["finalNrTPMovCorrs"] >> sm.finalNrTPMovCorrs;

    fs["finalNrTPStatCorrsFromLast"] >> sm.finalNrTPStatCorrsFromLast;

    fs["finalNrTPMovCorrsFromLast"] >> sm.finalNrTPMovCorrsFromLast;

    fs["finalNrTNStatCorrs"] >> sm.finalNrTNStatCorrs;

    fs["finalNrTNMovCorrs"] >> sm.finalNrTNMovCorrs;

    n = fs["combCorrsImg12TPorder"];
    n["statTPfromLast"] >> tmp;
    sm.combCorrsImg12TPorder.statTPfromLast = (unsigned char) tmp;
    n["statTPnew"] >> tmp;
    sm.combCorrsImg12TPorder.statTPnew = (unsigned char) tmp;
    n["movTPfromLast"] >> tmp;
    sm.combCorrsImg12TPorder.movTPfromLast = (unsigned char) tmp;
    n["movTPnew"] >> tmp;
    sm.combCorrsImg12TPorder.movTPnew = (unsigned char) tmp;

    fs["combCorrsImg12TNstatFirst"] >> tmp;
    sm.combCorrsImg12TNstatFirst = (tmp != 0);

    fs.release();

    return true;
}

bool readMultipleMatchSequencePars(const std::string &filename, std::vector<matchSequParameters> &matchPars){
    FileStorage fs = FileStorage(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }
    int nrEntries = 0;
    string parSetNr = "parSetNr";
    while(true) {
        cv::FileNode fn = fs[parSetNr + std::to_string(nrEntries)];
        if (fn.empty()) {
            break;
        }
        bool storePtClouds;
        fn["storePtClouds"] >> storePtClouds;
        if(!storePtClouds){
            cerr << "Point clouds were not stored when data was generated." << endl;
            fs.release();
            return false;
        }
        matchSequParameters pars;
        fn["hashMatchingPars"] >> pars.hashMatchingPars;
        fn["imgPath"] >> pars.imgPath;
        fn["imgPrePostFix"] >> pars.imgPrePostFix;
        fn["keyPointType"] >> pars.keyPointType;
        fn["descriptorType"] >> pars.descriptorType;
        fn["keypPosErrType"] >> pars.keypPosErrType;
        FileNode fn1 = fs["keypErrDistr"];
        double first_dbl = 0, second_dbl = 0;
        fn1["first"] >> first_dbl;
        fn1["second"] >> second_dbl;
        pars.keypErrDistr = make_pair(first_dbl, second_dbl);
        fn1 = fs["imgIntNoise"];
        fn1["first"] >> first_dbl;
        fn1["second"] >> second_dbl;
        pars.imgIntNoise = make_pair(first_dbl, second_dbl);
        fn1 = fs["repeatPatternPortStereo"];
        fn1["first"] >> first_dbl;
        fn1["second"] >> second_dbl;
        pars.repeatPatternPortStereo = make_pair(first_dbl, second_dbl);
        fn1 = fs["repeatPatternPortFToF"];
        fn1["first"] >> first_dbl;
        fn1["second"] >> second_dbl;
        pars.repeatPatternPortFToF = make_pair(first_dbl, second_dbl);
        fn["distortPatchCam1"] >> pars.distortPatchCam1;
        fn["oxfordGTMportion"] >> pars.oxfordGTMportion;
        fn["kittiGTMportion"] >> pars.kittiGTMportion;
        fn["megadepthGTMportion"] >> pars.megadepthGTMportion;
        fn["GTMportion"] >> pars.GTMportion;
        fn["WarpedPortionTN"] >> pars.WarpedPortionTN;
        fn["portionGrossTN"] >> pars.portionGrossTN;
        matchPars.emplace_back(move(pars));
        nrEntries++;
    }
    fs.release();
    return true;
}

bool readSequenceParameters(const std::string &filename, sequParameters &pars) {
    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }
    int tmp = 0;
    fs["nrStereoConfs"] >> tmp;
    pars.nrStereoConfs = (size_t)tmp;

    FileNode n = fs["inlRat"];
    if (n.type() != FileNode::SEQ) {
        cerr << "inlRat is not a sequence! FAIL" << endl;
        return false;
    }
    pars.inlRat.clear();
    FileNodeIterator it = n.begin(), it_end = n.end();
    while (it != it_end) {
        double inlRa1 = 0;
        it >> inlRa1;
        pars.inlRat.push_back(inlRa1);
    }

    fs["nFramesPerCamConf"] >> tmp;
    pars.nFramesPerCamConf = (size_t) tmp;

    n = fs["inlRatRange"];
    double first_dbl = 0, second_dbl = 0;
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    pars.inlRatRange = make_pair(first_dbl, second_dbl);

    fs["inlRatChanges"] >> pars.inlRatChanges;

    n = fs["truePosRange"];
    int first_int = 0, second_int = 0;
    n["first"] >> first_int;
    n["second"] >> second_int;
    pars.truePosRange = make_pair((size_t) first_int, (size_t) second_int);

    fs["truePosChanges"] >> pars.truePosChanges;

    fs["minKeypDist"] >> pars.minKeypDist;

    n = fs["corrsPerDepth"];
    n["near"] >> pars.corrsPerDepth.near;
    n["mid"] >> pars.corrsPerDepth.mid;
    n["far"] >> pars.corrsPerDepth.far;

    n = fs["corrsPerRegion"];
    if (n.type() != FileNode::SEQ) {
        cerr << "corrsPerRegion is not a sequence! FAIL" << endl;
        return false;
    }
    pars.corrsPerRegion.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        Mat m;
        it >> m;
        pars.corrsPerRegion.push_back(m.clone());
    }

    fs["corrsPerRegRepRate"] >> tmp;
    pars.corrsPerRegRepRate = (size_t) tmp;

    n = fs["depthsPerRegion"];
    if (n.type() != FileNode::SEQ) {
        cerr << "depthsPerRegion is not a sequence! FAIL" << endl;
        return false;
    }
    pars.depthsPerRegion = vector<vector<depthPortion>>(3, vector<depthPortion>(3));
    it = n.begin(), it_end = n.end();
    size_t idx = 0, x = 0, y = 0;
    for (; it != it_end; ++it) {
        y = idx / 3;
        x = idx % 3;

        FileNode n1 = *it;
        n1["near"] >> pars.depthsPerRegion[y][x].near;
        n1["mid"] >> pars.depthsPerRegion[y][x].mid;
        n1["far"] >> pars.depthsPerRegion[y][x].far;
        idx++;
    }

    n = fs["nrDepthAreasPReg"];
    if (n.type() != FileNode::SEQ) {
        cerr << "nrDepthAreasPReg is not a sequence! FAIL" << endl;
        return false;
    }
    pars.nrDepthAreasPReg = vector<vector<pair<size_t, size_t>>>(3, vector<pair<size_t, size_t>>(3));
    it = n.begin(), it_end = n.end();
    idx = 0;
    for (; it != it_end; ++it) {
        y = idx / 3;
        x = idx % 3;

        FileNode n1 = *it;
        n1["first"] >> first_int;
        n1["second"] >> second_int;
        pars.nrDepthAreasPReg[y][x] = make_pair((size_t) first_int, (size_t) second_int);
        idx++;
    }

    n = fs["camTrack"];
    if (n.type() != FileNode::SEQ) {
        cerr << "camTrack is not a sequence! FAIL" << endl;
        return false;
    }
    pars.camTrack.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        Mat m;
        it >> m;
        pars.camTrack.emplace_back(m.clone());
    }

    fs["relCamVelocity"] >> pars.relCamVelocity;

    fs["R"] >> pars.R;

    fs["nrMovObjs"] >> tmp;
    pars.nrMovObjs = (size_t) tmp;

    fs["startPosMovObjs"] >> pars.startPosMovObjs;

    n = fs["relAreaRangeMovObjs"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    pars.relAreaRangeMovObjs = make_pair(first_dbl, second_dbl);

    n = fs["movObjDepth"];
    if (n.type() != FileNode::SEQ) {
        cerr << "camTrack is not a sequence! FAIL" << endl;
        return false;
    }
    pars.movObjDepth.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        it >> tmp;
        pars.movObjDepth.push_back((depthClass) tmp);
    }

    fs["movObjDir"] >> pars.movObjDir;

    n = fs["relMovObjVelRange"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    pars.relMovObjVelRange = make_pair(first_dbl, second_dbl);

    fs["minMovObjCorrPortion"] >> pars.minMovObjCorrPortion;

    fs["CorrMovObjPort"] >> pars.CorrMovObjPort;

    fs["minNrMovObjs"] >> tmp;
    pars.minNrMovObjs = (size_t) tmp;

    n = fs["distortCamMat"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    pars.distortCamMat = make_pair(first_dbl, second_dbl);

    fs["totalNrFrames"] >> tmp;
    pars.totalNrFrames = (size_t) tmp;

    n = fs["nrCorrs"];
    if (n.type() != FileNode::SEQ) {
        cerr << "nrCorrs is not a sequence! FAIL" << endl;
        return false;
    }
    pars.nrCorrs.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        it >> tmp;
        pars.nrCorrs.push_back((size_t) tmp);
    }

    n = fs["absCamCoordinates"];
    if (n.type() != FileNode::SEQ) {
        cerr << "absCamCoordinates is not a sequence! FAIL" << endl;
        return false;
    }
    pars.absCamCoordinates.clear();
    it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        FileNode n1 = *it;
        Mat m1, m2;
        n1["R"] >> m1;
        n1["t"] >> m2;
        pars.absCamCoordinates.emplace_back(Poses(m1.clone(), m2.clone()));
    }

    n = fs["R_stereo"];
    if (n.type() != FileNode::SEQ) {
        cerr << "R_stereo is not a sequence! FAIL" << endl;
        return false;
    }
    pars.R_stereo.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        Mat R_stereo;
        it >> R_stereo;
        pars.R_stereo.emplace_back(R_stereo.clone());
    }

    n = fs["t_stereo"];
    if (n.type() != FileNode::SEQ) {
        cerr << "t_stereo is not a sequence! FAIL" << endl;
        return false;
    }
    pars.t_stereo.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        Mat t_stereo;
        it >> t_stereo;
        pars.t_stereo.emplace_back(t_stereo.clone());
    }

    fs["nrMovObjAllFrames"] >> tmp;
    pars.nrMovObjAllFrames = (size_t) tmp;

    //Read camera parameters
    fs["K1"] >> pars.K1;
    fs["K2"] >> pars.K2;

    n = fs["imgSize"];
    n["width"] >> first_int;
    n["height"] >> second_int;
    pars.imgSize = cv::Size(first_int, second_int);

    fs.release();

    return true;
}

void calcRelPose1(const cv::Mat &R_abs_1,
                  const cv::Mat &R_abs_2,
                  const cv::Mat &t_abs_1,
                  const cv::Mat &t_abs_2,
                  cv::Mat &R_rel,
                  cv::Mat &t_rel){
    R_rel = R_abs_2.t() * R_abs_1;
    t_rel = R_abs_2.t() * (t_abs_1 - t_abs_2);
}

void calcRelPose2(const cv::Mat &R_rel_1,
                  const cv::Mat &t_rel_1,
                  const cv::Mat &R_stereo_1,
                  const cv::Mat &R_stereo_2,
                  const cv::Mat &t_stereo_1,
                  const cv::Mat &t_stereo_2,
                  cv::Mat &R_rel,
                  cv::Mat &t_rel){
    R_rel = R_stereo_2 * R_rel_1 * R_stereo_1.t();
    t_rel = t_stereo_2 + R_stereo_2 * t_rel_1 - R_rel * t_stereo_1;
}

cv::Mat getEssentialMat(const cv::Mat &R_rel, const cv::Mat &t_rel){
    cv::Mat E = cv::Mat::zeros(3, 3, CV_64FC1);
    E.at<double>(0, 1) = -1. * t_rel.at<double>(2);
    E.at<double>(0, 2) = t_rel.at<double>(1);
    E.at<double>(1, 0) = t_rel.at<double>(2);
    E.at<double>(1, 2) = -1. * t_rel.at<double>(0);
    E.at<double>(2, 0) = -1. * t_rel.at<double>(1);
    E.at<double>(2, 1) = t_rel.at<double>(0);
    return E.clone();
}

cv::Mat getFundamentalMat(const cv::Mat &R_rel, const cv::Mat &t_rel, const cv::Mat &K1, const cv::Mat &K2){
    cv::Mat E = getEssentialMat(R_rel, t_rel);
    return K2.inv().t() * E * K1.inv();
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
void reOrderDescriptors(cv::Mat &descriptors, const std::vector<T> &idxs){
    CV_Assert((size_t)descriptors.rows == idxs.size());
    cv::Mat descriptor1_tmp;
    descriptor1_tmp.reserve(static_cast<size_t>(descriptors.rows));
    for(auto &idx : idxs){
        descriptor1_tmp.push_back(descriptors.row(static_cast<int>(idx)));
    }
    descriptor1_tmp.copyTo(descriptors);
}

bool getFrameToFrameMatches(const std::vector<int64_t> &combCorrsImg12TP_IdxWorld2_1,
                            const std::vector<int64_t> &combCorrsImg12TP_IdxWorld2_2,
                            const std::vector<size_t> &idxs_match23D_1,
                            const std::vector<size_t> &idxs_match23D_2,
                            const std::vector<cv::KeyPoint> &frameKeypoints_1,
                            const std::vector<cv::KeyPoint> &frameKeypoints_2,
                            const cv::Mat &frameDescriptors_1,
                            const cv::Mat &frameDescriptors_2,
                            const cv::Mat &K1,
                            const cv::Mat &K2,
                            const cv::Mat &R_rel,
                            const cv::Mat &t_rel,
                            const std::mt19937 &rand2,
                            std::vector<cv::KeyPoint> &matchKeypoints_1,
                            std::vector<cv::KeyPoint> &matchKeypoints_2,
                            cv::Mat &matchDescriptors_1,
                            cv::Mat &matchDescriptors_2,
                            std::vector<cv::DMatch> &matches){
    if(combCorrsImg12TP_IdxWorld2_1.empty() || combCorrsImg12TP_IdxWorld2_2.empty()){
        return false;
    }
    matches.clear();
    matchKeypoints_1.clear();
    matchKeypoints_2.clear();
    matchDescriptors_1.release();
    matchDescriptors_2.release();
    size_t nr_TP1 = combCorrsImg12TP_IdxWorld2_1.size();
    size_t nr_TP2 = combCorrsImg12TP_IdxWorld2_2.size();
    std::unordered_map<int64_t, size_t> idx3D1, idx3D2;
    for(size_t i = 0; i < idxs_match23D_1.size(); ++i){
        const size_t &id = idxs_match23D_1[i];
        if(id < nr_TP1){
            idx3D1[combCorrsImg12TP_IdxWorld2_1[id]] = i;
        }
    }
    for(size_t i = 0; i < idxs_match23D_2.size(); ++i){
        const size_t &id = idxs_match23D_2[i];
        if(id < nr_TP2){
            idx3D2[combCorrsImg12TP_IdxWorld2_1[id]] = i;
        }
    }
    matchDescriptors_1.reserve(nr_TP1);
    matchDescriptors_2.reserve(nr_TP2);
    matchKeypoints_1.reserve(nr_TP1);
    matchKeypoints_2.reserve(nr_TP2);
    std::unordered_set<size_t> corrs1, corrs2;
    std::unordered_map<int64_t, size_t>::iterator it;
    //Get TP
    for(auto &idx1: idx3D1){
        it = idx3D2.find(idx1.first);
        if(it != idx3D2.end()){
            corrs1.emplace(idx1.second);
            corrs2.emplace(it->second);
            matchKeypoints_1.push_back(frameKeypoints_1[idx1.second]);
            matchKeypoints_2.push_back(frameKeypoints_2[it->second]);
            matchDescriptors_1.push_back(frameDescriptors_1.row(static_cast<int>(idx1.second)));
            matchDescriptors_2.push_back(frameDescriptors_2.row(static_cast<int>(it->second)));
        }
    }
    if(matchKeypoints_1.empty()){
        return false;
    }
    cv::Mat F = getFundamentalMat(R_rel, t_rel, K1, K2);
    size_t cnt = 0;
    for(size_t i = 0; i < matchKeypoints_1.size(); i++){
        cv::Mat pt1 = (cv::Mat_<double>(3,1) << static_cast<double>(matchKeypoints_1[i].pt.x),
                static_cast<double>(matchKeypoints_1[i].pt.y), 1.);
        cv::Mat pt2 = (cv::Mat_<double>(1,3) << static_cast<double>(matchKeypoints_2[i].pt.x),
                static_cast<double>(matchKeypoints_2[i].pt.y), 1.);
        double err = abs(pt2.dot(F * pt1));
        if(err < 10.){
            cnt++;
        }
    }
    double err_rat = static_cast<double>(cnt) / static_cast<double>(matchKeypoints_1.size());
    if(err_rat < 0.6){
        return false;
    }
    //Get TN
    size_t minLen = min(idx3D1.size(), idx3D2.size()) - matchKeypoints_1.size();
    vector<size_t> tnidx1, tnidx2;
    for(size_t i = 0; i < idx3D1.size(); i++){
        if(corrs1.find(i) == corrs1.end()){
            tnidx1.push_back(i);
        }
    }
    for(size_t i = 0; i < idx3D2.size(); i++){
        if(corrs2.find(i) == corrs2.end()){
            tnidx2.push_back(i);
        }
    }
    if(!tnidx1.empty() && !tnidx2.empty()){
        std::shuffle(tnidx1.begin(), tnidx1.end(), rand2);
        std::shuffle(tnidx2.begin(), tnidx2.end(), rand2);
        for(size_t i = 0; i < minLen; i++){
            matchKeypoints_1.push_back(frameKeypoints_1[tnidx1[i]]);
            matchKeypoints_2.push_back(frameKeypoints_2[tnidx2[i]]);
            matchDescriptors_1.push_back(frameDescriptors_1.row(static_cast<int>(tnidx1[i])));
            matchDescriptors_2.push_back(frameDescriptors_2.row(static_cast<int>(tnidx2[i])));
        }
    }
    std::vector<std::size_t> shuffle_idx(matchKeypoints_1.size());
    std::iota(shuffle_idx.begin(), shuffle_idx.end(), 0);
    std::shuffle(shuffle_idx.begin(), shuffle_idx.end(), rand2);
    reOrderVector(matchKeypoints_1, shuffle_idx);
    reOrderVector(matchKeypoints_2, shuffle_idx);
    reOrderDescriptors(matchDescriptors_1, shuffle_idx);
    reOrderDescriptors(matchDescriptors_2, shuffle_idx);
    for(int i = 0; i < static_cast<int>(matchKeypoints_1.size()); i++) {
        cv::DMatch mtch;
        mtch.queryIdx = i;
        mtch.trainIdx = i;
        if (matchDescriptors_1.type() == CV_8U) {
            mtch.distance = static_cast<float>(cv::norm(matchDescriptors_1.row(i), matchDescriptors_2.row(i), cv::NORM_HAMMING));
        }else{
            mtch.distance = static_cast<float>(cv::norm(matchDescriptors_1.row(i), matchDescriptors_2.row(i), cv::NORM_L2));
        }
        matches.emplace_back(mtch);
    }
    return true;
}

bool getMultFrameToFrameMatches(const sequParameters &sequPars,
                                const std::vector<data3D> &sequData,
                                const std::vector<sequMatches> &matchData1,
                                std::vector<FrameToFrameMatches> &f2f_matches){
    for(size_t i = 1; i < sequPars.totalNrFrames; ++i) {
        FrameToFrameMatches f2f;
        cv::Mat R1, R2, t1, t2;
        size_t i1 = i - 1;
        calcRelPose1(sequPars.absCamCoordinates[i1].R,
                     sequPars.absCamCoordinates[i].R,
                     sequPars.absCamCoordinates[i1].t,
                     sequPars.absCamCoordinates[i].t,
                     R1,
                     t1);
        calcRelPose2(R1,
                     t1,
                     sequData[i1].actR,
                     sequData[i].actR,
                     sequData[i1].actT,
                     sequData[i].actT,
                     R2,
                     t2);
        std::random_device rd;
        std::mt19937 rand2(rd());
        std::vector<cv::KeyPoint> matchKeypoints1, matchKeypoints2;
        cv::Mat matchDescriptors1, matchDescriptors2;
        std::vector<cv::DMatch> matches;
        bool valid = false;
        if(getFrameToFrameMatches(sequData[i1].combCorrsImg12TP_IdxWorld2,
                                  sequData[i].combCorrsImg12TP_IdxWorld2,
                                  matchData1[i1].idxs_match23D1,
                                  matchData1[i].idxs_match23D1,
                                  matchData1[i1].frameKeypoints1,
                                  matchData1[i].frameKeypoints1,
                                  matchData1[i1].frameDescriptors1,
                                  matchData1[i].frameDescriptors1,
                                  sequPars.K1,
                                  sequPars.K2,
                                  R1,
                                  t1,
                                  rand2,
                                  matchKeypoints1,
                                  matchKeypoints2,
                                  matchDescriptors1,
                                  matchDescriptors2,
                                  matches)){
            f2f.frameKeypoints_1_1 = move(matchKeypoints1);
            f2f.frameKeypoints_2_1 = move(matchKeypoints2);
            matchDescriptors1.copyTo(f2f.descriptors_1_1);
            matchDescriptors2.copyTo(f2f.descriptors_2_1);
            f2f.matches_1 = move(matches);
            R1.copyTo(f2f.R_rel_1);
            t1.copyTo(f2f.t_rel_1);
            f2f.frameNr = i1;
            valid = true;
        }

        if(getFrameToFrameMatches(sequData[i1].combCorrsImg12TP_IdxWorld2,
                                  sequData[i].combCorrsImg12TP_IdxWorld2,
                                  matchData1[i1].idxs_match23D2,
                                  matchData1[i].idxs_match23D2,
                                  matchData1[i1].frameKeypoints2,
                                  matchData1[i].frameKeypoints2,
                                  matchData1[i1].frameDescriptors2,
                                  matchData1[i].frameDescriptors2,
                                  sequPars.K1,
                                  sequPars.K2,
                                  R2,
                                  t2,
                                  rand2,
                                  matchKeypoints1,
                                  matchKeypoints2,
                                  matchDescriptors1,
                                  matchDescriptors2,
                                  matches)){
            f2f.frameKeypoints_1_2 = move(matchKeypoints1);
            f2f.frameKeypoints_2_2 = move(matchKeypoints2);
            matchDescriptors1.copyTo(f2f.descriptors_1_2);
            matchDescriptors2.copyTo(f2f.descriptors_2_2);
            f2f.matches_2 = move(matches);
            R2.copyTo(f2f.R_rel_2);
            t2.copyTo(f2f.t_rel_2);
            f2f.frameNr = i1;
            valid = true;
        }
        if(valid){
            f2f_matches.emplace_back(move(f2f));
        }
    }

    return !f2f_matches.empty();
}



#endif //GENERATEVIRTUALSEQUENCE_LOADMATCHES_H
