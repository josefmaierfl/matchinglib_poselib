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
//
// Created by maierj on 5/14/20.
//

#ifndef GENERATEVIRTUALSEQUENCE_COLMAPBASE_H
#define GENERATEVIRTUALSEQUENCE_COLMAPBASE_H

#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <ceres/ceres.h>
#include "GTM/colmap/camera.h"
#include "GTM/colmap/camera_models.h"
#include "GTM/colmap/image.h"
#include "GTM/colmap/point3d.h"
//#ifdef CHECK
//#undef CHECK
//#endif
#include "GTM/colmap/logging.h"
#include "GTM/colmap/misc.h"
#include "GTM/colmap/math.h"
#include "GTM/colmap/pose.h"
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
    Eigen::Matrix3d R_rel;//Relative rotation matrix between images
    Eigen::Vector4d quat_rel;//Relative rotation quaternion between images (Refinement takes place on these values)
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

    bool getDepthMap2(){
        cv::Mat depthMap;
        if(!readDepthMap(depthImg2, depthMap)){
            return false;
        }
        cv::cv2eigen(depthMap, depthMap2);
        return true;
    }

    bool getDepthMap1(cv::Mat &depthMap1) const{
        return readDepthMap(depthImg1, depthMap1);
    }

    bool getKeypointsDepth(){
        cv::Mat depthMap;
        if(!readDepthMap(depthImg1, depthMap)){
            return false;
        }
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(4000);
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

    void getRelQuaternions(){
        quat_rel = RotationMatrixToQuaternion(R_rel);
    }

    void QuaternionToRotMat(){
        R_rel = QuaternionToRotationMatrix(quat_rel);
    }

    double calcReprojectionError(const std::pair<Eigen::Vector2i, double> &pt1, Eigen::Vector2d *loc2 = nullptr) const{
        Eigen::Vector2d pt1d = pt1.first.cast<double>();
        Eigen::Vector2d kpw1 = undistortedCam1.ImageToWorld(pt1d);
        Eigen::Vector3d p1(kpw1(0), kpw1(1), 1.);
        p1 *= pt1.second;
        p1 = R_rel * p1 + t_rel;
        p1 /= p1(2);
        kpw1 = undistortedCam2.WorldToImage(Eigen::Vector2d(p1(0), p1(1)));
        if(loc2){
            *loc2 = kpw1;
        }

        // Bilinear interpolation of depth in 2nd image
        double dtl, dtr, dbl, dbr;
        double xl = std::floor(kpw1(0));
        double xr = std::ceil(std::abs(kpw1(0)));
        double yt = std::floor(kpw1(1));
        double yb = std::ceil(std::abs(kpw1(1)));
        int xli = static_cast<int>(xl);
        int xri = static_cast<int>(xr);
        int yti = static_cast<int>(yt);
        int ybi = static_cast<int>(yb);
        if((xli < 0) || (xri >= depthMap2.cols()) || (yti < 0) || (ybi >= depthMap2.rows())){
            return DBL_MAX;
        }
        dtl = depthMap2(yti, xli);
        dtr = depthMap2(yti, xri);
        dbl = depthMap2(ybi, xli);
        dbr = depthMap2(ybi, xri);
        vector<double*> depths;
        vector<size_t> validD, nValid;
        depths.push_back(&dtl);
        depths.push_back(&dtr);
        depths.push_back(&dbl);
        depths.push_back(&dbr);
        size_t idx = 0;
        for(auto &d: depths){
            if(*d < 1e-3){
                nValid.push_back(idx++);
            }else{
                validD.push_back(idx++);
            }
        }
        if(validD.empty()){
            return DBL_MAX;
        }
        if(!nValid.empty()){
            double meanD = 0;
            for(auto &i: validD){
                meanD += *(depths[i]);
            }
            meanD /= static_cast<double>(validD.size());
            for(auto &i: nValid){
                *(depths[i]) = meanD;
            }
        }
        double xdiff = xr - xl;
        double dt, db, d2;
        if(xdiff < 1e-4){
            dt = dtl;
            db = dbl;
        }else {
            double xrDiff = xr - kpw1(0);
            double xlDiff = kpw1(0) - xl;
            dt = ((dtl * xrDiff + dtr * xlDiff)) / xdiff;
            db = ((dbl * xrDiff + dbr * xlDiff)) / xdiff;
        }
        double ydiff = yb - yt;
        if(ydiff < 1e-4){
            d2 = dt;
        }else {
            d2 = ((dt * (yb - kpw1(1)) + db * (kpw1(1) - yt))) / ydiff;
        }

        p1 *= d2;
        p1 -= t_rel;
        p1 = R_rel.transpose() * p1;
        p1 /= p1(2);
        kpw1 = undistortedCam1.WorldToImage(Eigen::Vector2d(p1(0), p1(1)));
        kpw1 -= pt1d;
        return kpw1.squaredNorm();
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

struct BundleAdjustmentOptions {
    // Loss function types: Trivial (non-robust) and Cauchy (robust) loss.
    enum class LossFunctionType { TRIVIAL, SOFT_L1, CAUCHY, HUBER };
    LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

    // Scaling factor determines residual at which robustification takes place.
    double loss_function_scale = 1.0;

    // Whether to refine the focal length parameter group.
//    bool refine_focal_length = true;

    // Whether to refine the principal point parameter group.
//    bool refine_principal_point = false;

    // Whether to refine the extra parameter group.
//    bool refine_extra_params = true;

    // Whether to refine the extrinsic parameter group.
//    bool refine_extrinsics = true;

    // Whether to print a final summary.
    bool print_summary = true;

    int CeresCPUcnt = -1;

    // Minimum number of residuals to enable multi-threading. Note that
    // single-threaded is typically better for small bundle adjustment problems
    // due to the overhead of threading.
//    int min_num_residuals_for_multi_threading = 50000;

    // Ceres-Solver options.
    ceres::Solver::Options solver_options;

    BundleAdjustmentOptions() {
        solver_options.function_tolerance = 0.0;
        solver_options.gradient_tolerance = 0.0;
        solver_options.parameter_tolerance = 0.0;
        solver_options.minimizer_progress_to_stdout = false;
        solver_options.max_num_iterations = 100;
        solver_options.max_linear_solver_iterations = 200;
        solver_options.max_num_consecutive_invalid_steps = 10;
        solver_options.max_consecutive_nonmonotonic_steps = 10;
        solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
        solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
    }

    // Create a new loss function based on the specified options. The caller
    // takes ownership of the loss function.
    ceres::LossFunction* CreateLossFunction() const;

    bool Check() const;
};

class RigFixedDepthBundleAdjuster{
public:
    explicit RigFixedDepthBundleAdjuster(BundleAdjustmentOptions options);
    ~RigFixedDepthBundleAdjuster(){
        problem_.release();
        delete loss_function;
    }

    bool Solve(corrStats* corrImgs);

private:
    void SetUp(corrStats* corrImgs);

    void AddImageToProblem(corrStats* corrImgs);

    void ParameterizeCameraRigs();

    const double max_reproj_error = 50.0;

    const BundleAdjustmentOptions options_;
    std::unique_ptr<ceres::Problem> problem_;
    ceres::LossFunction* loss_function = nullptr;
    ceres::Solver::Summary summary_;

    // The Quaternions added to the problem, used to set the local
    // parameterization once after setting up the problem.
    std::unordered_set<double*> parameterized_qvec_data_;
};

// Return the number of logical CPU cores if num_threads <= 0,
// otherwise return the input value of num_threads.
int GetEffectiveNumThreads(const int num_threads);

void PrintSolverSummary(const ceres::Solver::Summary& summary);

class colmapBase{
public:
    bool prepareColMapData(const megaDepthFolders& folders);
    bool getFileNames(const megaDepthFolders& folders);
    explicit colmapBase(bool refineSfM_ = true, bool storeFlowFile_ = true, uint32_t verbose_ = 0, int CeresCPUcnt_ = -1):
    num_added_points3D_(0),
    min_num_points3D_Img_(100),
    maxdepthImgSize(0),
    refineSfM(refineSfM_),
    storeFlowFile(storeFlowFile_),
    verbose(verbose_),
    CeresCPUcnt(CeresCPUcnt_){}
    bool calculateFlow(const std::string &flowPath, std::vector<megaDepthData> &data);
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
    bool getCorrectDims();
    static bool checkScale(const size_t &dimWsrc, const int &dimWdest, const size_t &dimHsrc, const int &dimHdest);
    static bool estimateFlow(cv::Mat &flow, const corrStats &data);

    // Get const objects.
//    inline const colmap::Camera& Camera(const camera_t camera_id) const;
//    inline const colmap::Image& Image(const image_t image_id) const;

    // Get mutable objects.
    inline Camera& getCamera(const camera_t camera_id);
    inline Image& getImage(const image_t image_id);

    EIGEN_STL_UMAP(camera_t, class Camera) cameras_;
    EIGEN_STL_UMAP(image_t, class Image) images_;
    EIGEN_STL_UMAP(point3D_t, class Point3D) points3D_;
    EIGEN_STL_UMAP(image_t, struct corrStats) correspImgs_;
    std::vector<image_t> reg_image_ids_;
    // Total number of added 3D points, used to generate unique identifiers.
    point3D_t num_added_points3D_;
    point2D_t min_num_points3D_Img_;
    int maxdepthImgSize;
    bool refineSfM;
    bool storeFlowFile;
    uint32_t verbose = 0;
    int CeresCPUcnt;
};

//const colmap::Camera& colmapBase::Camera(const camera_t camera_id) const {
//    return cameras_.at(camera_id);
//}
//
//const colmap::Image& colmapBase::Image(const image_t image_id) const {
//    return images_.at(image_id);
//}

Camera& colmapBase::getCamera(const camera_t camera_id) {
    return cameras_.at(camera_id);
}

Image& colmapBase::getImage(const image_t image_id) {
    return images_.at(image_id);
}

#endif //GENERATEVIRTUALSEQUENCE_COLMAPBASE_H
