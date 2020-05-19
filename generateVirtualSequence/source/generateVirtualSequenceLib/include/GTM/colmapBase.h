//
// Created by maierj on 5/14/20.
//

#ifndef GENERATEVIRTUALSEQUENCE_COLMAPBASE_H
#define GENERATEVIRTUALSEQUENCE_COLMAPBASE_H

#include <unordered_map>
#include <functional>
#include "GTM/colmap/camera.h"
#include "GTM/colmap/camera_models.h"
#include "GTM/colmap/image.h"
#include "GTM/colmap/point3d.h"
#include "GTM/colmap/logging.h"
#include "GTM/colmap/misc.h"
#include "GTM/colmap/math.h"
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "GTM/prepareMegaDepth.h"
#include "helper_funcs.h"
#include "opencv2/hdf.hpp"
#include <opencv2/core/eigen.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace colmap;
using namespace std;

struct corrStats{
    image_t imgID;//Image ID of matching image
    size_t nrCorresp3D;//Number of 3D coordinates found equal by SfM for both images
    double viewAngle;//View angle in radians between the 2 images
    double tvecNorm;//Length of relative translation vector between images
    double weight;//(nrCorresp3D / 10)^2 * viewAngle * tvecNorm
    double scale;//Scale between image used within SfM and image with corresponding dense depth maps
    Eigen::Vector2d shift;//Shift between image used within SfM and image with corresponding dense depth maps
    string depthImg1;//Path and name of the depth map of img1
    string depthImg2;//Path and name of the depth map of img2
    string imgOrig1;//Original first image used in SfM
    string imgOrig2;//Original second image used in SfM
    string img1;//Scaled and shifted image of imgOrig1
    string img2;//Scaled and shifted image of imgOrig2
    cv::Size imgSize1;//Dimensions of img1
    cv::Size imgSize2;//Dimensions of img1
    Camera undistortedCam1;//Camera parameters for the undistorted and scaled camera of img1
    Camera undistortedCam2;//Camera parameters for the undistorted and scaled camera of img2
    Eigen::Matrix3d R_rel;//Relative rotation matrix  between images
    Eigen::Vector3d t_rel;//Relative translation vector between images
    Eigen::MatrixXd depthMap2;//Depth map of the second image
    std::vector<std::pair<Eigen::Vector2i, double>> keypDepth1;//Keypoints and corresponding depth values in the first image

    corrStats(){
        imgID = 0;
        nrCorresp3D = 0;
        scale = 1.;
        shift = Eigen::Vector2d(0, 0);
        viewAngle = 0;
        tvecNorm = 0;
        weight = 0;
    }

    corrStats(image_t &imgID_, size_t nrCorresp3D_, double &viewAngle_, double &tvecNorm_, Eigen::Matrix3d R_rel_, Eigen::Vector3d t_rel_):
            imgID(imgID_),
            nrCorresp3D(nrCorresp3D_),
            viewAngle(viewAngle_),
            tvecNorm(tvecNorm_),
            weight(0),
            scale(1.),
            shift(0,0),
            R_rel(move(R_rel_)),
            t_rel(move(t_rel_)){}

    corrStats(image_t &imgID_, size_t &nrCorresp3D_, double &viewAngle_, double &tvecNorm_,
            double &weight_, Eigen::Matrix3d R_rel_, Eigen::Vector3d t_rel_):
            imgID(imgID_),
            nrCorresp3D(nrCorresp3D_),
            viewAngle(viewAngle_),
            tvecNorm(tvecNorm_),
            weight(weight_),
            scale(1.),
            shift(0,0),
            R_rel(move(R_rel_)),
            t_rel(move(t_rel_)){}

    static bool readDepthMap(const std::string &filename, cv::Mat &depth){
        cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(filename);
        if(!h5io->hlexists("/depth")){
            h5io->close();
            return false;
        }
        h5io->dsread(depth, "/depth");
        h5io->close();
        if(depth.channels() > 1){
            return false;
        }
        depth.convertTo(depth, CV_64FC1);
        return cv::countNonZero(depth < 0) == 0;
    }

    bool getKeypointsDepth(){
        cv::Mat depthMap;
        if(!readDepthMap(depthImg1, depthMap)){
            return false;
        }
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(1000);
        if(detector.empty()){
            cerr << "Cannot create keypoint detector!" << endl;
            return false; //Error creating feature detector
        }
        cv::Mat img = cv::imread(img1, cv::IMREAD_GRAYSCALE);
        vector<cv::KeyPoint> keypoints;
        detector->detect( img, keypoints );
        keypDepth1.clear();
        double maxHeight = static_cast<double>(undistortedCam2.Height()) - 10.;
        double maxWidth = static_cast<double>(undistortedCam2.Width()) - 10.;
        for(auto &kp: keypoints){
            Eigen::Vector2d kp1(round(static_cast<double>(kp.pt.x)), round(static_cast<double>(kp.pt.y)));
            Eigen::Vector2d kp2 = undistortedCam1.ImageToWorld(kp1);
            Eigen::Vector3d pt(kp2(0), kp2(1), 1.);
            Eigen::Vector2i kpi(static_cast<int>(kp1(0)), static_cast<int>(kp1(1)));
            const double d = depthMap.at<double>(kpi(1), kpi(0));
            pt *= d;
            pt = R_rel * pt + t_rel;
            pt /= pt(2);
            kp2 = undistortedCam2.WorldToImage(Eigen::Vector2d(pt(0), pt(1)));
            if(kp2(0) < 10. || kp2(0) > maxWidth || kp2(1) < 10. || kp2(1) > maxHeight){
                continue;
            }
            keypDepth1.emplace_back(move(kpi), d);
        }
        return keypDepth1.size() > 100;
    }
};

struct UndistortCameraOptions {
    // The amount of blank pixels in the undistorted image in the range [0, 1].
    double blank_pixels = 0.0;

    // Minimum and maximum scale change of camera used to satisfy the blank
    // pixel constraint.
    double min_scale = 0.2;
    double max_scale = 2.0;

    // Maximum image size in terms of width or height of the undistorted camera.
    int max_image_size = -1;

    // The 4 factors in the range [0, 1] that define the ROI (region of interest)
    // in original image. The bounding box pixel coordinates are calculated as
    //    (roi_min_x * Width, roi_min_y * Height) and
    //    (roi_max_x * Width, roi_max_y * Height).
    double roi_min_x = 0.0;
    double roi_min_y = 0.0;
    double roi_max_x = 1.0;
    double roi_max_y = 1.0;
};

class colmapBase{
public:
    bool prepareColMapData(const megaDepthFolders& folders);
    bool getFileNames(const megaDepthFolders& folders);
    colmapBase(): num_added_points3D_(0), min_num_points3D_Img_(100), maxdepthImgSize(0){}
private:
    void getMaxDepthImgDim();
    void getUndistortedScaledDims();
    void ReadText(const std::string& path);
    void filterExistingDepth(const megaDepthFolders& folders);
    void filterInterestingImgs();
    bool getCorrespondingImgs();
    void ReadCamerasText(const std::string& path);
    void ReadImagesText(const std::string& path);
    void ReadPoints3DText(const std::string& path);
    Camera UndistortCamera(const UndistortCameraOptions& options,
                           const Camera& camera);
    bool refineRelPoses();
    bool checkCorrectDimensions();
    static bool checkScale(const size_t &dimWsrc, const int &dimWdest, const size_t &dimHsrc, const int &dimHdest);

    // Get const objects.
    inline const class Camera& Camera(const camera_t camera_id) const;
    inline const class Image& Image(const image_t image_id) const;

    // Get mutable objects.
    inline class Camera& Camera(const camera_t camera_id);
    inline class Image& Image(const image_t image_id);

    EIGEN_STL_UMAP(camera_t, class Camera) cameras_;
    EIGEN_STL_UMAP(image_t, class Image) images_;
    EIGEN_STL_UMAP(point3D_t, class Point3D) points3D_;
    EIGEN_STL_UMAP(image_t, struct corrStats) correspImgs_;
    std::vector<image_t> reg_image_ids_;
    // Total number of added 3D points, used to generate unique identifiers.
    point3D_t num_added_points3D_;
    point2D_t min_num_points3D_Img_;
    int maxdepthImgSize;
};

const class Camera& colmapBase::Camera(const camera_t camera_id) const {
    return cameras_.at(camera_id);
}

const class Image& colmapBase::Image(const image_t image_id) const {
    return images_.at(image_id);
}

class Camera& colmapBase::Camera(const camera_t camera_id) {
    return cameras_.at(camera_id);
}

class Image& colmapBase::Image(const image_t image_id) {
    return images_.at(image_id);
}

#endif //GENERATEVIRTUALSEQUENCE_COLMAPBASE_H
