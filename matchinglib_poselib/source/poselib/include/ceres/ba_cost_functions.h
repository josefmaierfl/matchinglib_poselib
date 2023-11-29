//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2021 Josef Maier
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

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <functional>

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "ceres/types_colmap.h"
#include "ceres/camera_models.h"
#include "ceres/camera.h"
#include "ceres/pose.h"

#include "poselib/pose_helper.h"

// #include <utils_common.h>
// #include <utils_cv.h>

namespace poselib
{
    struct StereoBAData
    {
        colmap::Camera cam1;
        colmap::Camera cam2;
        Eigen::Matrix3d R_rel;                                          //Relative rotation matrix between images
        Eigen::Vector4d quat_rel;                                       //Relative rotation quaternion between images (Refinement takes place on these values)
        Eigen::Vector3d t_rel;                                          //Relative translation vector between images
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> corrs; //Correspondences
        std::vector<Eigen::Vector3d> points3d;                          // 3D points
        std::vector<size_t> constant_points3d;                          // Indices of 3D points that should be kept constant
        bool inCamCoordinates = false; //If true, corrs are assumed in the camera coordinate system (no camera calibration parameters required)
        bool dist_damp = false; //Specifies if a damping term should be added to residuals for limiting distortion coefficients
        std::vector<colmap::CamMatDampingSingle> camMatDampPars; // Parameters for damping camera matrix coeffitients (vector must be of size 2)
        int camMat_damp = colmap::CamMatDampApply::NO_DAMPING;   // Specifies for which camera parameters damping should be applied

        void getRelQuaternions()
        {
            quat_rel = colmap::RotationMatrixToQuaternion(R_rel);
        }

        void QuaternionToRotMat()
        {
            R_rel = colmap::QuaternionToRotationMatrix(quat_rel);
        }

        double calcReprojectionError(const Eigen::Vector2d &pt_img1, const Eigen::Vector2d &pt_img2, const Eigen::Vector3d &pt_3d) const
        {
            Eigen::Vector2d pw1, pw2;
            Eigen::Vector3d pw13, pw23;

            // Rotate and translate.
            Eigen::Vector3d pw2d = colmap::QuaternionRotatePoint(quat_rel, pt_3d);
            pw2d += t_rel;

            pw13 = pt_3d / pt_3d.z();
            pw23 = pw2d / pw2d.z();
            if(inCamCoordinates){
                pw1 = Eigen::Vector2d(pw13(0), pw13(1));
                pw2 = Eigen::Vector2d(pw23(0), pw23(1));
            }
            else
            {
                pw1 = cam1.WorldToImage(Eigen::Vector2d(pw13(0), pw13(1)));
                pw2 = cam2.WorldToImage(Eigen::Vector2d(pw23(0), pw23(1)));
            }
            Eigen::Vector2d pw1_err = pw1 - pt_img1;
            Eigen::Vector2d pw2_err = pw2 - pt_img2;
            const double mean_err = (pw1_err.squaredNorm() + pw2_err.squaredNorm()) / 2.;
            return mean_err;
        }
    };

    struct GlobalBAData
    {
        std::vector<colmap::Camera> cams;                       // Different cameras (i.e. sharing not the same intrinsics)
        std::vector<size_t> imgs_to_cam_idx;                    // Indices pointing to cams for each image (multiple images can share the same camera)
        std::vector<bool> constant_cams;                        // Indicates which cameras (i.e. intrinsics) should be kept constant
        std::vector<Eigen::Matrix3d> Rs;                         //Absolute rotation matrix for each image
        std::vector<Eigen::Vector4d> quats;                      //Absolute rotation quaternion for each image (Refinement takes place on these values)
        std::vector<Eigen::Vector3d> ts;                         //Absolute translation vector for each image
        std::vector<bool> constant_poses;                        // Indicates which poses (i.e. R & t) should be kept constant
        std::vector<std::vector<Eigen::Vector2d>> corrs;         // Image projections
        std::vector<Eigen::Vector3d> points3d;                   // 3D points
        std::vector<size_t> constant_points3d;                   // Indices of 3D points that should be kept constant
        std::vector<std::vector<size_t>> points3d_idx;           //Indices to points3d for every image projection of each camera
        bool dist_damp = false;                                  //Specifies if a damping term should be added to residuals for limiting distortion coefficients
        std::vector<colmap::CamMatDampingSingle> camMatDampPars; // Parameters for damping camera matrix coeffitients (vector must be of same size as cams)
        int camMat_damp = colmap::CamMatDampApply::NO_DAMPING;   // Specifies for which camera parameters damping should be applied

        void getRelQuaternions()
        {
            for (auto &R : Rs){
                quats.emplace_back(colmap::RotationMatrixToQuaternion(R));
            }
        }

        void QuaternionToRotMat()
        {
            for (size_t i = 0; i < quats.size(); ++i){
                Rs[i] = colmap::QuaternionToRotationMatrix(quats[i]);
            }
        }

        bool Check() const
        {
            if(cams.empty()){
                std::cerr << "No cams provided." << std::endl;
                return false;
            }
            const size_t max_cams = cams.size();
            const size_t max_imgs = corrs.size();
            if (!constant_cams.empty() && constant_cams.size() != max_cams){
                std::cerr << "Invalid constant_cams size" << std::endl;
                return false;
            }
            if (imgs_to_cam_idx.size() != max_imgs || Rs.size() != max_imgs || ts.size() != max_imgs || quats.size() != max_imgs || (!constant_poses.empty() && constant_poses.size() != max_imgs) || points3d_idx.size() != max_imgs){
                std::cerr << "Invalid size of one or more image information containers" << std::endl;
                return false;
            }
            for (const auto &i : imgs_to_cam_idx){
                if (i >= max_cams){
                    std::cerr << "Invalid imgs_to_cam_idx entry" << std::endl;
                    return false;
                }
            }
            for (size_t i = 0; i < max_imgs; ++i){
                if (corrs.at(i).size() != points3d_idx.at(i).size()){
                    std::cerr << "Invalid size of one or more points3d_idx entries" << std::endl;
                    return false;
                }
            }
            const size_t max_3dpts = points3d.size();
            if (!constant_points3d.empty()){
                for (const auto &i : constant_points3d){
                    if(i >= max_3dpts){
                        std::cerr << "Invalid constant_points3d entry" << std::endl;
                        return false;
                    }
                }
            }
            if (camMat_damp)
            {
                if (camMatDampPars.size() != max_cams){
                    std::cerr << "Number of camera mat damping objects must be equal to number of cameras" << std::endl;
                    return false;
                }
                for (const auto &cd : camMatDampPars){
                    if ((camMat_damp & colmap::CamMatDampApply::DAMP_F_CHANGE) && (nearZero(cd.fChangeMax) || cd.fChangeMax < 0))
                    {
                        std::cerr << "Invalid damping parameter fChangeMax" << std::endl;
                        return false;
                    }
                    if ((camMat_damp & colmap::CamMatDampApply::DAMP_FX_FY_RATIO) && (nearZero(cd.fxfyRatioDiffMax) || cd.fxfyRatioDiffMax < 0))
                    {
                        std::cerr << "Invalid damping parameter fxfyRatioDiffMax" << std::endl;
                        return false;
                    }
                    if ((camMat_damp & colmap::CamMatDampApply::DAMP_CX_CY) && (nearZero(cd.cxcyDistMidRatioMax) || cd.cxcyDistMidRatioMax < 0))
                    {
                        std::cerr << "Invalid damping parameter cxcyDistMidRatioMax" << std::endl;
                        return false;
                    }
                }
            }
            return true;
        }

        double calcReprojectionError(const Eigen::Vector2d &pt_img, const Eigen::Vector3d &pt_3d, const Eigen::Vector4d &quat, const Eigen::Vector3d &t, const colmap::Camera &cam) const
        {
            // Rotate and translate.
            Eigen::Vector3d pw2d = colmap::QuaternionRotatePoint(quat, pt_3d);
            pw2d += t;

            pw2d /= pw2d.z();
            Eigen::Vector2d pw2 = cam.WorldToImage(Eigen::Vector2d(pw2d(0), pw2d(1)));

            Eigen::Vector2d pw2_err = pw2 - pt_img;
            return pw2_err.squaredNorm();
        }
    };

    struct RelativePoseBAData
    {
        std::unordered_map<int, colmap::Camera> cams;            // Different cameras (i.e. sharing not the same intrinsics)
        std::unordered_map<int, bool> constant_cams;                  // Indicates which cameras (i.e. intrinsics) should be kept constant
        std::unordered_map<std::pair<int, int>, Eigen::Matrix3d, pair_hash, pair_EqualTo> Rs; //Relative rotation matrix for each image pair
        std::unordered_map<std::pair<int, int>, Eigen::Vector4d, pair_hash, pair_EqualTo> quats; //Relative rotation quaternion for each image pair (Refinement takes place on these values)
        std::unordered_map<std::pair<int, int>, Eigen::Vector3d, pair_hash, pair_EqualTo> ts;    //Relative translation vector for each image pair
        std::unordered_map<std::pair<int, int>, bool, pair_hash, pair_EqualTo> constant_poses;   // Indicates which poses (i.e. R & t) should be kept constant
        std::unordered_map<std::pair<int, int>, std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>, pair_hash, pair_EqualTo> corrs; // Image projections
        bool dist_damp = false;                                                                                                                        //Specifies if a damping term should be added to residuals for limiting distortion coefficients
        std::unordered_map<int, colmap::CamMatDampingSingle> camMatDampPars;                                                                           // Parameters for damping camera matrix coeffitients (vector must be of same size as cams)
        int camMat_damp = colmap::CamMatDampApply::NO_DAMPING;                                                                                         // Specifies for which camera parameters damping should be applied

        RelativePoseBAData(const std::unordered_map<int, colmap::Camera> &cams_, const std::unordered_map<std::pair<int, int>, Eigen::Matrix3d, pair_hash, pair_EqualTo> &Rs_, const std::unordered_map<std::pair<int, int>, Eigen::Vector3d, pair_hash, pair_EqualTo> &ts_, const std::unordered_map<std::pair<int, int>, std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>, pair_hash, pair_EqualTo> &corrs_, const bool distortion_damping = false) : cams(cams_), Rs(Rs_), ts(ts_), corrs(corrs_), dist_damp(distortion_damping)
        {
            for (auto &t : ts)
            {
                t.second /= t.second.norm();
            }
            getRelQuaternions();
        }

        void getRelQuaternions()
        {
            for (auto &R : Rs)
            {
                quats.emplace(R.first, colmap::RotationMatrixToQuaternion(R.second));
            }
        }

        void overwritePose(const std::pair<int, int> &idx, const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
        {
            ts.at(idx) = t / t.norm();
            Rs.at(idx) = R;
            quats.at(idx) = colmap::RotationMatrixToQuaternion(R);
        }

        bool getRE(const std::pair<int, int> &idx, Eigen::Matrix3d &E, Eigen::Matrix3d &R)
        {
            bool transpose_E = false;
            std::pair<int, int> ii12 = idx;
            if (Rs.find(ii12) == Rs.end())
            {
                transpose_E = true;
                ii12 = std::make_pair(idx.second, idx.first);
                if (Rs.find(ii12) == Rs.end())
                {
                    return false;
                }
            }
            R = Rs.at(ii12);
            E = getEfromRT(R, ts.at(ii12));
            if (transpose_E)
            {
                E.transposeInPlace();
                R.transposeInPlace();
            }
            return true;
        }

        void QuaternionToRotMat()
        {
            for (auto &q : quats)
            {
                Rs.at(q.first) = colmap::QuaternionToRotationMatrix(q.second);
            }
        }

        bool Check() const
        {
            if (cams.empty())
            {
                std::cerr << "No cams provided." << std::endl;
                return false;
            }
            const size_t max_cams = cams.size();
            const size_t max_imgs = corrs.size();
            if (!constant_cams.empty() && constant_cams.size() != max_cams)
            {
                std::cerr << "Invalid constant_cams size" << std::endl;
                return false;
            }
            if (Rs.size() != max_imgs || ts.size() != max_imgs || quats.size() != max_imgs || (!constant_poses.empty() && constant_poses.size() != max_imgs))
            {
                std::cerr << "Invalid size of one or more image information containers" << std::endl;
                return false;
            }
            for (auto &m : corrs)
            {
                if(m.second.first.size() != m.second.second.size()){
                    std::cerr << "Number of correspondences must match" << std::endl;
                    return false;
                }
                if (Rs.find(m.first) == Rs.end() || ts.find(m.first) == ts.end() || quats.find(m.first) == quats.end()){
                    std::cerr << "Missing relative poses of pair " << m.first.first << "-" << m.first.second << std::endl;
                    return false;
                }
                if (cams.find(m.first.first) == cams.end() || cams.find(m.first.second) == cams.end()){
                    std::cerr << "Missing camera parameters of cam " << m.first.first << " or " << m.first.second << std::endl;
                    return false;
                }
            }
            if (camMat_damp)
            {
                if (camMatDampPars.size() != max_cams)
                {
                    std::cerr << "Number of camera mat damping objects must be equal to number of cameras" << std::endl;
                    return false;
                }
                for (const auto &cd : camMatDampPars)
                {
                    if ((camMat_damp & colmap::CamMatDampApply::DAMP_F_CHANGE) && (nearZero(cd.second.fChangeMax) || cd.second.fChangeMax < 0))
                    {
                        std::cerr << "Invalid damping parameter fChangeMax" << std::endl;
                        return false;
                    }
                    if ((camMat_damp & colmap::CamMatDampApply::DAMP_FX_FY_RATIO) && (nearZero(cd.second.fxfyRatioDiffMax) || cd.second.fxfyRatioDiffMax < 0))
                    {
                        std::cerr << "Invalid damping parameter fxfyRatioDiffMax" << std::endl;
                        return false;
                    }
                    if ((camMat_damp & colmap::CamMatDampApply::DAMP_CX_CY) && (nearZero(cd.second.cxcyDistMidRatioMax) || cd.second.cxcyDistMidRatioMax < 0))
                    {
                        std::cerr << "Invalid damping parameter cxcyDistMidRatioMax" << std::endl;
                        return false;
                    }
                }
            }
            return true;
        }

        double calcReprojectionError(const Eigen::Vector2d &pt_img1, const Eigen::Vector2d &pt_img2, const Eigen::Vector4d &quat, const Eigen::Vector3d &t, const colmap::Camera &cam1, const colmap::Camera &cam2) const
        {
            Eigen::Matrix3d R = colmap::QuaternionToRotationMatrix(quat);

            // Matrix representation of the cross product t x R.
            Eigen::Matrix3d t_x;
            t_x << 0., -t(2), t(1), t(2), 0., -t(0), -t(1), t(0), 0.;

            // Essential matrix.
            const Eigen::Matrix3d E = t_x * R;

            // Undistort and transform to camera coordinates.
            Eigen::Vector2d pc1 = cam1.ImageToWorld(pt_img1);
            Eigen::Vector2d pc2 = cam2.ImageToWorld(pt_img2);

            // Homogeneous image coordinates.
            const Eigen::Vector3d x1_h(pc1(0), pc1(1), 1.);
            const Eigen::Vector3d x2_h(pc2(0), pc2(1), 1.);

            // Squared sampson error.
            const Eigen::Vector3d Ex1 = E * x1_h;
            const Eigen::Vector3d Etx2 = E.transpose() * x2_h;
            const double x2tEx1 = x2_h.transpose() * Ex1;
            const double err = x2tEx1 * x2tEx1 /
                               (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                                Etx2(1) * Etx2(1));
            return err;
        }
    };

    struct FixedDepthBAData
    {
        std::vector<colmap::Camera> cams;                   // Different cameras (i.e. sharing not the same intrinsics)
        std::vector<size_t> imgs_to_cam_idx;                // Indices pointing to cams for each image (multiple images can share the same camera)
        std::vector<bool> constant_cams;                    // Indicates which cameras (i.e. intrinsics) should be kept constant
        std::vector<Eigen::Matrix3d> Rs;                    //Absolute rotation matrix for each image
        std::vector<Eigen::Vector4d> quats;                 //Absolute rotation quaternion for each image (Refinement takes place on these values)
        std::vector<Eigen::Vector3d> ts;                    //Absolute translation vector for each image
        std::vector<bool> constant_poses;                   // Indicates which poses (i.e. R & t) should be kept constant
        std::vector<std::vector<Eigen::Vector2d>> corrs;    // Image projections
        std::vector<Eigen::Vector3d> points3d;              // 3D points
        std::unordered_map<int, Eigen::Vector2d> depth_scales_imgs;           //Scale (x) and additive term (y) for each image (not cam) in depth_vals_imgs
        const std::vector<std::pair<int, double>> depth_vals_imgs;                  //Depth of 3D points and img number. Order corresponds to points3d.
        std::vector<std::vector<size_t>> points3d_idx;      //Indices to points3d for every image projection of each camera
        bool dist_damp = false;                             //Specifies if a damping term should be added to residuals for limiting distortion coefficients
        std::vector<colmap::CamMatDampingSingle> camMatDampPars; // Parameters for damping camera matrix coeffitients (vector must be of same size as cams)
        int camMat_damp = colmap::CamMatDampApply::NO_DAMPING;   // Specifies for which camera parameters damping should be applied

        FixedDepthBAData(const std::vector<colmap::Camera> &cams_, const std::vector<size_t> &imgs_to_cam_idx_, const std::vector<Eigen::Matrix3d> &Rs_, const std::vector<Eigen::Vector3d> &ts_, const std::vector<std::vector<Eigen::Vector2d>> &corrs_, const std::vector<Eigen::Vector3d> &points3d_, const std::vector<std::pair<int, double>> &depth_vals_imgs_, const std::unordered_map<int, Eigen::Vector2d> &depth_scales_imgs_, const std::vector<std::vector<size_t>> &points3d_idx_, const bool distortion_damping = false) : cams(cams_), imgs_to_cam_idx(imgs_to_cam_idx_), Rs(Rs_), ts(ts_), corrs(corrs_), points3d(points3d_), depth_scales_imgs(depth_scales_imgs_), depth_vals_imgs(depth_vals_imgs_), points3d_idx(points3d_idx_), dist_damp(distortion_damping)
        {
            quats.clear();
            getRelQuaternions();
        }

        void getRelQuaternions()
        {
            for (auto &R : Rs)
            {
                quats.emplace_back(colmap::RotationMatrixToQuaternion(R));
            }
        }

        void QuaternionToRotMat()
        {
            for (size_t i = 0; i < quats.size(); ++i)
            {
                Rs[i] = colmap::QuaternionToRotationMatrix(quats[i]);
            }
        }

        bool Check() const
        {
            if (cams.empty())
            {
                std::cerr << "No cams provided." << std::endl;
                return false;
            }
            const size_t max_cams = cams.size();
            const size_t max_imgs = corrs.size();
            if (!constant_cams.empty() && constant_cams.size() != max_cams)
            {
                std::cerr << "Invalid constant_cams size" << std::endl;
                return false;
            }
            if (imgs_to_cam_idx.size() != max_imgs || Rs.size() != max_imgs || ts.size() != max_imgs || quats.size() != max_imgs || (!constant_poses.empty() && constant_poses.size() != max_imgs) || points3d_idx.size() != max_imgs)
            {
                std::cerr << "Invalid size of one or more image information containers" << std::endl;
                return false;
            }
            for (const auto &i : imgs_to_cam_idx)
            {
                if (i >= max_cams)
                {
                    std::cerr << "Invalid imgs_to_cam_idx entry" << std::endl;
                    return false;
                }
            }
            if (corrs.at(0).size() != depth_vals_imgs.size())
            {
                std::cerr << "Invalid number of depth values for 1st cam" << std::endl;
                return false;
            }
            for(const auto &d : depth_vals_imgs){
                if (depth_scales_imgs.find(d.first) == depth_scales_imgs.end()){
                    std::cerr << "Missing depth scale parameter for image " << d.first << std::endl;
                    return false;
                }
            }
            for (size_t i = 0; i < max_imgs; ++i)
            {
                if (corrs.at(i).size() != points3d_idx.at(i).size())
                {
                    std::cerr << "Invalid size of one or more points3d_idx or planeParam_idx entries" << std::endl;
                    return false;
                }
            }
            if (camMat_damp)
            {
                if (camMatDampPars.size() != max_cams)
                {
                    std::cerr << "Number of camera mat damping objects must be equal to number of cameras" << std::endl;
                    return false;
                }
                for (const auto &cd : camMatDampPars)
                {
                    if ((camMat_damp & colmap::CamMatDampApply::DAMP_F_CHANGE) && (nearZero(cd.fChangeMax) || cd.fChangeMax < 0))
                    {
                        std::cerr << "Invalid damping parameter fChangeMax" << std::endl;
                        return false;
                    }
                    if ((camMat_damp & colmap::CamMatDampApply::DAMP_FX_FY_RATIO) && (nearZero(cd.fxfyRatioDiffMax) || cd.fxfyRatioDiffMax < 0))
                    {
                        std::cerr << "Invalid damping parameter fxfyRatioDiffMax" << std::endl;
                        return false;
                    }
                    if ((camMat_damp & colmap::CamMatDampApply::DAMP_CX_CY) && (nearZero(cd.cxcyDistMidRatioMax) || cd.cxcyDistMidRatioMax < 0))
                    {
                        std::cerr << "Invalid damping parameter cxcyDistMidRatioMax" << std::endl;
                        return false;
                    }
                }
            }
            return true;
        }

        double calcReprojectionError(const Eigen::Vector2d &pt_img, const Eigen::Vector3d &pt_3d, const double &depth_fixed, const Eigen::Vector2d &depth_scale_add, const Eigen::Vector4d &quat, const Eigen::Vector3d &t, const colmap::Camera &cam) const
        {
            //Calculate 3D point at fixed depth
            Eigen::Vector3d pt_3d_fd = pt_3d;
            double z = depth_scale_add.x() * depth_fixed + depth_scale_add.y();
            pt_3d_fd /= pt_3d_fd.z();
            pt_3d_fd *= z;

            // Rotate and translate.
            Eigen::Vector3d pw2d = colmap::QuaternionRotatePoint(quat, pt_3d_fd);
            pw2d += t;

            pw2d /= pw2d.z();
            Eigen::Vector2d pw2 = cam.WorldToImage(Eigen::Vector2d(pw2d(0), pw2d(1)));

            Eigen::Vector2d pw2_err = pw2 - pt_img;
            return pw2_err.squaredNorm();
        }
    };

    // Standard bundle adjustment cost function for variable
    // camera pose and calibration and point parameters.
    template <typename CameraModel>
    class BundleAdjustmentCostFunction
    {
    public:
        explicit BundleAdjustmentCostFunction(const Eigen::Vector2d &point2D, const bool distortion_damping = false, const colmap::CamMatDampingSingle *camMat_damp = nullptr)
            : observed_x_(point2D(0)), observed_y_(point2D(1)), dist_damp(distortion_damping), cam_damp(camMat_damp) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &point2D, const bool distortion_damping = false, const colmap::CamMatDampingSingle *camMat_damp = nullptr)
        {
            return (new ceres::AutoDiffCostFunction<
                    BundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 3,
                    CameraModel::kNumParams>(
                new BundleAdjustmentCostFunction(point2D, distortion_damping, camMat_damp)));
        }

        template <typename T>
        bool operator()(const T *const qvec, const T *const tvec,
                        const T *const point3D, const T *const camera_params,
                        T *residuals) const
        {
            // Rotate and translate.
            T projection[3];
            ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
            projection[0] += tvec[0];
            projection[1] += tvec[1];
            projection[2] += tvec[2];

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // Distort and transform to pixel space.
            CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                      &residuals[0], &residuals[1]);

            // Re-projection error.
            residuals[0] -= T(observed_x_);
            residuals[1] -= T(observed_y_);

            if(dist_damp){
                CameraModel::AddDistortionDamping(camera_params, &residuals[0], &residuals[1]);
            }

            if (cam_damp)
            {
                if (cam_damp->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params, &(cam_damp->focalMid), &(cam_damp->focalRange2), &(cam_damp->f_init), &(cam_damp->fChangeMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params, &(cam_damp->fxfyRatioDiffMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params, &(cam_damp->cx_init), &(cam_damp->cy_init), &(cam_damp->imgDiag_2), &(cam_damp->cxcyDistMidRatioMax), &residuals[0], &residuals[1]);
                }
            }

            return true;
        }

    private:
        const double observed_x_;
        const double observed_y_;
        const bool dist_damp;
        const colmap::CamMatDampingSingle *cam_damp;
    };

    // Bundle adjustment cost function for variable
    // camera calibration and point parameters, and fixed camera pose.
    template <typename CameraModel>
    class BundleAdjustmentConstantPoseCostFunction
    {
    public:
        BundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d &qvec,
                                                 const Eigen::Vector3d &tvec,
                                                 const Eigen::Vector2d &point2D,
                                                 const bool distortion_damping = false,
                                                 const colmap::CamMatDampingSingle *camMat_damp = nullptr)
            : qw_(qvec(0)),
              qx_(qvec(1)),
              qy_(qvec(2)),
              qz_(qvec(3)),
              tx_(tvec(0)),
              ty_(tvec(1)),
              tz_(tvec(2)),
              observed_x_(point2D(0)),
              observed_y_(point2D(1)),
              dist_damp(distortion_damping),
              cam_damp(camMat_damp) {}

        static ceres::CostFunction *Create(const Eigen::Vector4d &qvec,
                                           const Eigen::Vector3d &tvec,
                                           const Eigen::Vector2d &point2D,
                                           const bool distortion_damping = false,
                                           const colmap::CamMatDampingSingle *camMat_damp = nullptr)
        {
            return (new ceres::AutoDiffCostFunction<
                    BundleAdjustmentConstantPoseCostFunction<CameraModel>, 2, 3,
                    CameraModel::kNumParams>(
                new BundleAdjustmentConstantPoseCostFunction(qvec, tvec, point2D, distortion_damping, camMat_damp)));
        }

        template <typename T>
        bool operator()(const T *const point3D, const T *const camera_params,
                        T *residuals) const
        {
            const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

            // Rotate and translate.
            T projection[3];
            ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
            projection[0] += T(tx_);
            projection[1] += T(ty_);
            projection[2] += T(tz_);

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // Distort and transform to pixel space.
            CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                      &residuals[0], &residuals[1]);

            // Re-projection error.
            residuals[0] -= T(observed_x_);
            residuals[1] -= T(observed_y_);

            if (dist_damp)
            {
                CameraModel::AddDistortionDamping(camera_params, &residuals[0], &residuals[1]);
            }

            if (cam_damp)
            {
                if (cam_damp->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params, &(cam_damp->focalMid), &(cam_damp->focalRange2), &(cam_damp->f_init), &(cam_damp->fChangeMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params, &(cam_damp->fxfyRatioDiffMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params, &(cam_damp->cx_init), &(cam_damp->cy_init), &(cam_damp->imgDiag_2), &(cam_damp->cxcyDistMidRatioMax), &residuals[0], &residuals[1]);
                }
            }

            return true;
        }

    private:
        const double qw_;
        const double qx_;
        const double qy_;
        const double qz_;
        const double tx_;
        const double ty_;
        const double tz_;
        const double observed_x_;
        const double observed_y_;
        const bool dist_damp;
        const colmap::CamMatDampingSingle *cam_damp;
    };

    // Rig bundle adjustment cost function for variable camera pose and calibration
    // and point parameters. Different from the standard bundle adjustment function,
    // this cost function is suitable for camera rigs with consistent relative poses
    // of the cameras within the rig. The cost function first projects points into
    // the local system of the camera rig and then into the local system of the
    // camera within the rig.
    template <typename CameraModel>
    class RigBundleAdjustmentCostFunction
    {
    public:
        explicit RigBundleAdjustmentCostFunction(const Eigen::Vector2d &point2D, const bool distortion_damping = false, const colmap::CamMatDampingSingle *camMat_damp = nullptr)
            : observed_x_(point2D(0)), observed_y_(point2D(1)), dist_damp(distortion_damping), cam_damp(camMat_damp) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &point2D, const bool distortion_damping = false, const colmap::CamMatDampingSingle *camMat_damp = nullptr)
        {
            return (new ceres::AutoDiffCostFunction<
                    RigBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 4, 3, 3,
                    CameraModel::kNumParams>(
                new RigBundleAdjustmentCostFunction(point2D, distortion_damping, camMat_damp)));
        }

        template <typename T>
        bool operator()(const T *const rig_qvec, const T *const rig_tvec,
                        const T *const rel_qvec, const T *const rel_tvec,
                        const T *const point3D, const T *const camera_params,
                        T *residuals) const
        {
            // Concatenate rotations.
            T qvec[4];
            ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

            // Concatenate translations.
            T tvec[3];
            ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
            tvec[0] += rel_tvec[0];
            tvec[1] += rel_tvec[1];
            tvec[2] += rel_tvec[2];

            // Rotate and translate.
            T projection[3];
            ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
            projection[0] += tvec[0];
            projection[1] += tvec[1];
            projection[2] += tvec[2];

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // Distort and transform to pixel space.
            CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                      &residuals[0], &residuals[1]);

            // Re-projection error.
            residuals[0] -= T(observed_x_);
            residuals[1] -= T(observed_y_);

            if (dist_damp)
            {
                CameraModel::AddDistortionDamping(camera_params, &residuals[0], &residuals[1]);
            }

            if (cam_damp)
            {
                if (cam_damp->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params, &(cam_damp->focalMid), &(cam_damp->focalRange2), &(cam_damp->f_init), &(cam_damp->fChangeMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params, &(cam_damp->fxfyRatioDiffMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params, &(cam_damp->cx_init), &(cam_damp->cy_init), &(cam_damp->imgDiag_2), &(cam_damp->cxcyDistMidRatioMax), &residuals[0], &residuals[1]);
                }
            }

            return true;
        }

    private:
        const double observed_x_;
        const double observed_y_;
        const bool dist_damp;
        const colmap::CamMatDampingSingle *cam_damp;
    };

    // Stereo bundle adjustment cost function for variable relative camera pose and calibration
    // and point parameters.
    template <typename CameraModel>
    class StereoBundleAdjustmentCostFunction
    {
    public:
        explicit StereoBundleAdjustmentCostFunction(const Eigen::Vector2d &point2D_img1, const Eigen::Vector2d &point2D_img2, const bool distortion_damping = false, const colmap::CamMatDampingSingle *camMat_damp1 = nullptr, const colmap::CamMatDampingSingle *camMat_damp2 = nullptr)
            : observed_x1_(point2D_img1(0)), observed_y1_(point2D_img1(1)), observed_x2_(point2D_img2(0)), observed_y2_(point2D_img2(1)), dist_damp(distortion_damping), cam_damp1(camMat_damp1), cam_damp2(camMat_damp2) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &point2D_img1, const Eigen::Vector2d &point2D_img2, const bool distortion_damping = false, const colmap::CamMatDampingSingle *camMat_damp1 = nullptr, const colmap::CamMatDampingSingle *camMat_damp2 = nullptr)
        {
            return (new ceres::AutoDiffCostFunction<
                    StereoBundleAdjustmentCostFunction<CameraModel>, 4, 4, 3, 3,
                    CameraModel::kNumParams, CameraModel::kNumParams>(
                new StereoBundleAdjustmentCostFunction(point2D_img1, point2D_img2, distortion_damping, camMat_damp1, camMat_damp2)));
        }

        template <typename T>
        bool operator()(const T *const rel_qvec, const T *const rel_tvec,
                        const T *const point3D, const T *const camera_params1, const T *const camera_params2,
                        T *residuals) const
        {
            // Project to first image plane.
            T projection1[2];
            projection1[0] = point3D[0] / point3D[2];
            projection1[1] = point3D[1] / point3D[2];

            // Distort and transform to pixel space.
            CameraModel::WorldToImage(camera_params1, projection1[0], projection1[1],
                                      &residuals[0], &residuals[1]);

            // Re-projection error.
            residuals[0] -= T(observed_x1_);
            residuals[1] -= T(observed_y1_);

            // Rotate and translate.
            T projection2[3];
            ceres::UnitQuaternionRotatePoint(rel_qvec, point3D, projection2);
            projection2[0] += rel_tvec[0];
            projection2[1] += rel_tvec[1];
            projection2[2] += rel_tvec[2];

            // Project to second image plane.
            projection2[0] /= projection2[2];
            projection2[1] /= projection2[2];

            // Distort and transform to pixel space.
            CameraModel::WorldToImage(camera_params2, projection2[0], projection2[1],
                                      &residuals[2], &residuals[3]);

            // Re-projection error.
            residuals[2] -= T(observed_x2_);
            residuals[3] -= T(observed_y2_);

            if (dist_damp)
            {
                CameraModel::AddDistortionDamping(camera_params1, &residuals[0], &residuals[1]);
                CameraModel::AddDistortionDamping(camera_params2, &residuals[2], &residuals[3]);
            }

            if (cam_damp1)
            {
                if (cam_damp1->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params1, &(cam_damp1->focalMid), &(cam_damp1->focalRange2), &(cam_damp1->f_init), &(cam_damp1->fChangeMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp1->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params1, &(cam_damp1->fxfyRatioDiffMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp1->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params1, &(cam_damp1->cx_init), &(cam_damp1->cy_init), &(cam_damp1->imgDiag_2), &(cam_damp1->cxcyDistMidRatioMax), &residuals[0], &residuals[1]);
                }
            }
            if (cam_damp2)
            {
                if (cam_damp2->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params2, &(cam_damp2->focalMid), &(cam_damp2->focalRange2), &(cam_damp2->f_init), &(cam_damp2->fChangeMax), &residuals[2], &residuals[3]);
                }
                if (cam_damp2->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params2, &(cam_damp2->fxfyRatioDiffMax), &residuals[2], &residuals[3]);
                }
                if (cam_damp2->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params2, &(cam_damp2->cx_init), &(cam_damp2->cy_init), &(cam_damp2->imgDiag_2), &(cam_damp2->cxcyDistMidRatioMax), &residuals[2], &residuals[3]);
                }
            }

            return true;
        }

    private:
        const double observed_x1_;
        const double observed_y1_;
        const double observed_x2_;
        const double observed_y2_;
        const bool dist_damp;
        const colmap::CamMatDampingSingle *cam_damp1;
        const colmap::CamMatDampingSingle *cam_damp2;
    };

    // Stereo bundle adjustment cost function for variable relative camera pose and point parameters without calibration parameters
    class StereoBundleAdjustmentCostFunctionCamCoordinates
    {
    public:
        explicit StereoBundleAdjustmentCostFunctionCamCoordinates(const Eigen::Vector2d &point2D_img1, const Eigen::Vector2d &point2D_img2)
            : observed_x1_(point2D_img1(0)), observed_y1_(point2D_img1(1)), observed_x2_(point2D_img2(0)), observed_y2_(point2D_img2(1)) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &point2D_img1, const Eigen::Vector2d &point2D_img2)
        {
            return (new ceres::AutoDiffCostFunction<
                    StereoBundleAdjustmentCostFunctionCamCoordinates, 4, 4, 3, 3>(
                new StereoBundleAdjustmentCostFunctionCamCoordinates(point2D_img1, point2D_img2)));
        }

        template <typename T>
        bool operator()(const T *const rel_qvec, const T *const rel_tvec,
                        const T *const point3D,
                        T *residuals) const
        {
            // Project to first image plane.
            residuals[0] = point3D[0] / point3D[2];
            residuals[1] = point3D[1] / point3D[2];

            // Re-projection error.
            residuals[0] -= T(observed_x1_);
            residuals[1] -= T(observed_y1_);

            // Rotate and translate.
            T projection2[3];
            ceres::UnitQuaternionRotatePoint(rel_qvec, point3D, projection2);
            projection2[0] += rel_tvec[0];
            projection2[1] += rel_tvec[1];
            projection2[2] += rel_tvec[2];

            // Project to second image plane.
            residuals[2] = projection2[0] / projection2[2];
            residuals[3] = projection2[1] / projection2[2];

            // Re-projection error.
            residuals[2] -= T(observed_x2_);
            residuals[3] -= T(observed_y2_);

            return true;
        }

    private:
        const double observed_x1_;
        const double observed_y1_;
        const double observed_x2_;
        const double observed_y2_;
    };

    // Cost function for refining two-view geometry based on the Sampson-Error.
    //
    // First pose is assumed to be located at the origin with 0 rotation. Second
    // pose is assumed to be on the unit sphere around the first pose, i.e. the
    // pose of the second camera is parameterized by a 3D rotation and a
    // 3D translation with unit norm. `tvec` is therefore over-parameterized as is
    // and should be down-projected using `HomogeneousVectorParameterization`.
    // Correspondences must be in the camera coordinate system, i.e. normalized by
    // camera matrices
    class RelativePoseCostFunction
    {
    public:
        RelativePoseCostFunction(const Eigen::Vector2d &x1, const Eigen::Vector2d &x2)
            : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &x1,
                                           const Eigen::Vector2d &x2)
        {
            return (new ceres::AutoDiffCostFunction<RelativePoseCostFunction, 1, 4, 3>(
                new RelativePoseCostFunction(x1, x2)));
        }

        template <typename T>
        bool operator()(const T *const qvec, const T *const tvec,
                        T *residuals) const
        {
            Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
            ceres::QuaternionToRotation(qvec, R.data());

            // Matrix representation of the cross product t x R.
            Eigen::Matrix<T, 3, 3> t_x;
            t_x << T(0), -tvec[2], tvec[1], tvec[2], T(0), -tvec[0], -tvec[1], tvec[0],
                T(0);

            // Essential matrix.
            const Eigen::Matrix<T, 3, 3> E = t_x * R;

            // Homogeneous image coordinates.
            const Eigen::Matrix<T, 3, 1> x1_h(T(x1_), T(y1_), T(1));
            const Eigen::Matrix<T, 3, 1> x2_h(T(x2_), T(y2_), T(1));

            // Squared sampson error.
            const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
            const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
            const T x2tEx1 = x2_h.transpose() * Ex1;
            residuals[0] = x2tEx1 * x2tEx1 /
                           (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                            Etx2(1) * Etx2(1));

            return true;
        }

    private:
        const double x1_;
        const double y1_;
        const double x2_;
        const double y2_;
    };

    // Cost function for refining two-view geometry based on the Sampson-Error.
    //
    // First pose is assumed to be located at the origin with 0 rotation. Second
    // pose is assumed to be on the unit sphere around the first pose, i.e. the
    // pose of the second camera is parameterized by a 3D rotation and a
    // 3D translation with unit norm. `tvec` is therefore over-parameterized as is
    // and should be down-projected using `HomogeneousVectorParameterization`.
    // Correspondences must be in the image coordinate system
    template <typename CameraModel>
    class RelativePoseImgCoordsCostFunction
    {
    public:
        RelativePoseImgCoordsCostFunction(const Eigen::Vector2d &x1, const Eigen::Vector2d &x2, const bool distortion_damping = false, const colmap::CamMatDampingSingle *camMat_damp1 = nullptr, const colmap::CamMatDampingSingle *camMat_damp2 = nullptr)
            : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)), dist_damp(distortion_damping), cam_damp1(camMat_damp1), cam_damp2(camMat_damp2) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &x1,
                                           const Eigen::Vector2d &x2,
                                           const bool distortion_damping = false,
                                           const colmap::CamMatDampingSingle *camMat_damp1 = nullptr,
                                           const colmap::CamMatDampingSingle *camMat_damp2 = nullptr)
        {
            return (new ceres::AutoDiffCostFunction<RelativePoseImgCoordsCostFunction, 1, 4, 3, CameraModel::kNumParams, CameraModel::kNumParams>(
                new RelativePoseImgCoordsCostFunction(x1, x2, distortion_damping, camMat_damp1, camMat_damp2)));
        }

        template <typename T>
        bool operator()(const T *const qvec, const T *const tvec, const T *const camera_params1, const T *const camera_params2, T *residuals) const
        {
            Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
            ceres::QuaternionToRotation(qvec, R.data());

            // Matrix representation of the cross product t x R.
            Eigen::Matrix<T, 3, 3> t_x;
            t_x << T(0), -tvec[2], tvec[1], tvec[2], T(0), -tvec[0], -tvec[1], tvec[0],
                T(0);

            // Essential matrix.
            const Eigen::Matrix<T, 3, 3> E = t_x * R;

            // Undistort and transform to camera coordinates.
            T x1_cam[2], x2_cam[2];
            CameraModel::ImageToWorld(camera_params1, T(x1_), T(y1_),
                                      &x1_cam[0], &x1_cam[1]);
            CameraModel::ImageToWorld(camera_params2, T(x2_), T(y2_),
                                      &x2_cam[0], &x2_cam[1]);

            // Homogeneous image coordinates.
            const Eigen::Matrix<T, 3, 1> x1_h(x1_cam[0], x1_cam[1], T(1.));
            const Eigen::Matrix<T, 3, 1> x2_h(x2_cam[0], x2_cam[1], T(1.));

            // Squared sampson error.
            const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
            const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
            const T x2tEx1 = x2_h.transpose() * Ex1;
            residuals[0] = x2tEx1 * x2tEx1 /
                           (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                            Etx2(1) * Etx2(1));

            T res_sign, resPlus;
            if (dist_damp || cam_damp1 || cam_damp2)
            {
                res_sign = residuals[0] / (ceres::abs(residuals[0]) + T(1e-9));
                if (res_sign < T(1e-2) && res_sign > T(-1e-2)){
                    res_sign = T(1.0);
                }
                resPlus = res_sign * T(1e-9) + residuals[0];
            }

            if (dist_damp)
            {
                T res_save[2], tmp = T(0);
                res_save[0] = residuals[0];
                res_save[1] = residuals[0];
                CameraModel::AddDistortionDamping(camera_params1, &res_save[0], &tmp);
                CameraModel::AddDistortionDamping(camera_params2, &res_save[1], &tmp);
                const T damp1 = res_save[0] / resPlus;
                const T damp2 = res_save[1] / resPlus;
                residuals[0] *= damp1 * damp2;
                if (cam_damp1 || cam_damp2)
                {
                    resPlus = res_sign * T(1e-9) + residuals[0];
                }
            }

            if (cam_damp1 && cam_damp1->damping)
            {
                T tmp = T(0);
                T res_save = residuals[0];
                if (cam_damp1->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params1, &(cam_damp1->focalMid), &(cam_damp1->focalRange2), &(cam_damp1->f_init), &(cam_damp1->fChangeMax), &res_save, &tmp);
                }
                if (cam_damp1->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params1, &(cam_damp1->fxfyRatioDiffMax), &res_save, &tmp);
                }
                if (cam_damp1->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params1, &(cam_damp1->cx_init), &(cam_damp1->cy_init), &(cam_damp1->imgDiag_2), &(cam_damp1->cxcyDistMidRatioMax), &res_save, &tmp);
                }
                const T damp1 = res_save / resPlus;
                residuals[0] *= damp1;
                resPlus = res_sign * T(1e-9) + residuals[0];
            }

            if (cam_damp2 && cam_damp2->damping)
            {
                T tmp = T(0);
                T res_save = residuals[0];
                if (cam_damp2->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params2, &(cam_damp2->focalMid), &(cam_damp2->focalRange2), &(cam_damp2->f_init), &(cam_damp2->fChangeMax), &res_save, &tmp);
                }
                if (cam_damp2->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params2, &(cam_damp2->fxfyRatioDiffMax), &res_save, &tmp);
                }
                if (cam_damp2->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params2, &(cam_damp2->cx_init), &(cam_damp2->cy_init), &(cam_damp2->imgDiag_2), &(cam_damp2->cxcyDistMidRatioMax), &res_save, &tmp);
                }
                const T damp2 = res_save / resPlus;
                residuals[0] *= damp2;
            }

            return true;
        }

    private:
        const double x1_;
        const double y1_;
        const double x2_;
        const double y2_;
        const bool dist_damp;
        const colmap::CamMatDampingSingle *cam_damp1;
        const colmap::CamMatDampingSingle *cam_damp2;
    };

    // Cost function for refining two-view geometry based on the Sampson-Error.
    //
    // First pose is assumed to be located at the origin with 0 rotation. Second
    // pose is assumed to be on the unit sphere around the first pose, i.e. the
    // pose of the second camera is parameterized by a 3D rotation and a
    // 3D translation with unit norm. `tvec` is therefore over-parameterized as is
    // and should be down-projected using `HomogeneousVectorParameterization`.
    // Correspondences must be in the image coordinate system
    template <typename CameraModel>
    class RelativeConstPoseCostFunction
    {
    public:
        RelativeConstPoseCostFunction(const Eigen::Vector4d &qvec,
                                      const Eigen::Vector3d &tvec,
                                      const Eigen::Vector2d &x1,
                                      const Eigen::Vector2d &x2,
                                      const bool distortion_damping = false,
                                      const colmap::CamMatDampingSingle *camMat_damp1 = nullptr,
                                      const colmap::CamMatDampingSingle *camMat_damp2 = nullptr)
            : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)), E_(Rt2E(qvec, tvec)), dist_damp(distortion_damping), cam_damp1(camMat_damp1), cam_damp2(camMat_damp2) {}

        static ceres::CostFunction *Create(const Eigen::Vector4d &qvec,
                                           const Eigen::Vector3d &tvec,
                                           const Eigen::Vector2d &x1,
                                           const Eigen::Vector2d &x2,
                                           const bool distortion_damping = false,
                                           const colmap::CamMatDampingSingle *camMat_damp1 = nullptr,
                                           const colmap::CamMatDampingSingle *camMat_damp2 = nullptr)
        {
            return (new ceres::AutoDiffCostFunction<RelativeConstPoseCostFunction, 1, CameraModel::kNumParams, CameraModel::kNumParams>(
                new RelativeConstPoseCostFunction(qvec, tvec, x1, x2, distortion_damping, camMat_damp1, camMat_damp2)));
        }

        template <typename T>
        bool operator()(const T *const camera_params1, const T *const camera_params2, T *residuals) const
        {
            // Essential matrix.
            const Eigen::Matrix<T, 3, 3> E = E_.cast<T>();

            // Undistort and transform to camera coordinates.
            T x1_cam[2], x2_cam[2];
            CameraModel::ImageToWorld(camera_params1, T(x1_), T(y1_),
                                      &x1_cam[0], &x1_cam[1]);
            CameraModel::ImageToWorld(camera_params2, T(x2_), T(y2_),
                                      &x2_cam[0], &x2_cam[1]);

            // Homogeneous image coordinates.
            const Eigen::Matrix<T, 3, 1> x1_h(x1_cam[0], x1_cam[1], T(1.));
            const Eigen::Matrix<T, 3, 1> x2_h(x2_cam[0], x2_cam[1], T(1.));

            // Squared sampson error.
            const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
            const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
            const T x2tEx1 = x2_h.transpose() * Ex1;
            residuals[0] = x2tEx1 * x2tEx1 /
                           (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                            Etx2(1) * Etx2(1));

            T res_sign, resPlus;
            if (dist_damp || cam_damp1 || cam_damp2)
            {
                res_sign = residuals[0] / (ceres::abs(residuals[0]) + T(1e-9));
                if (res_sign < T(1e-2) && res_sign > T(-1e-2))
                {
                    res_sign = T(1.0);
                }
                resPlus = res_sign * T(1e-9) + residuals[0];
            }

            if (dist_damp)
            {
                T res_save[2], tmp = T(0);
                res_save[0] = residuals[0];
                res_save[1] = residuals[0];
                CameraModel::AddDistortionDamping(camera_params1, &res_save[0], &tmp);
                CameraModel::AddDistortionDamping(camera_params2, &res_save[1], &tmp);
                const T damp1 = res_save[0] / resPlus;
                const T damp2 = res_save[1] / resPlus;
                residuals[0] *= damp1 * damp2;
                if (cam_damp1 || cam_damp2)
                {
                    resPlus = res_sign * T(1e-9) + residuals[0];
                }
            }

            if (cam_damp1 && cam_damp1->damping)
            {
                T tmp = T(0);
                T res_save = residuals[0];
                if (cam_damp1->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params1, &(cam_damp1->focalMid), &(cam_damp1->focalRange2), &(cam_damp1->f_init), &(cam_damp1->fChangeMax), &res_save, &tmp);
                }
                if (cam_damp1->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params1, &(cam_damp1->fxfyRatioDiffMax), &res_save, &tmp);
                }
                if (cam_damp1->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params1, &(cam_damp1->cx_init), &(cam_damp1->cy_init), &(cam_damp1->imgDiag_2), &(cam_damp1->cxcyDistMidRatioMax), &res_save, &tmp);
                }
                const T damp1 = res_save / resPlus;
                residuals[0] *= damp1;
                resPlus = res_sign * T(1e-9) + residuals[0];
            }

            if (cam_damp2 && cam_damp2->damping)
            {
                T tmp = T(0);
                T res_save = residuals[0];
                if (cam_damp2->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params2, &(cam_damp2->focalMid), &(cam_damp2->focalRange2), &(cam_damp2->f_init), &(cam_damp2->fChangeMax), &res_save, &tmp);
                }
                if (cam_damp2->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params2, &(cam_damp2->fxfyRatioDiffMax), &res_save, &tmp);
                }
                if (cam_damp2->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params2, &(cam_damp2->cx_init), &(cam_damp2->cy_init), &(cam_damp2->imgDiag_2), &(cam_damp2->cxcyDistMidRatioMax), &res_save, &tmp);
                }
                const T damp2 = res_save / resPlus;
                residuals[0] *= damp2;
            }

            return true;
        }

    private:
        Eigen::Matrix3d Rt2E(const Eigen::Vector4d &qvec,
                             const Eigen::Vector3d &tvec)
        {
            const Eigen::Matrix3d R = colmap::QuaternionToRotationMatrix(qvec);

            // Matrix representation of the cross product t x R.
            Eigen::Matrix3d t_x;
            t_x << 0., -tvec(2), tvec(1), tvec(2), 0., -tvec(0), -tvec(1), tvec(0), 0.;

            // Essential matrix.
            return t_x * R;
        }
        const double x1_;
        const double y1_;
        const double x2_;
        const double y2_;
        const Eigen::Matrix3d E_;
        const bool dist_damp;
        const colmap::CamMatDampingSingle *cam_damp1;
        const colmap::CamMatDampingSingle *cam_damp2;
    };

    // Bundle adjustment cost function using fixed 3D point depths (all depth values can change linearly): variable
    // camera calibration, point parameters, global depth scale & additive term, and camera pose.
    template <typename CameraModel>
    class BAFixedDepthCostFunction
    {
    public:
        BAFixedDepthCostFunction(const double &depth,
                                 const Eigen::Vector2d &point2D,
                                 const bool distortion_damping = false,
                                 const colmap::CamMatDampingSingle *camMat_damp = nullptr)
            : depth_(depth),
              observed_x_(point2D(0)),
              observed_y_(point2D(1)),
              dist_damp(distortion_damping),
              cam_damp(camMat_damp) {}

        static ceres::CostFunction *Create(const double &depth,
                                           const Eigen::Vector2d &point2D,
                                           const bool distortion_damping = false,
                                           const colmap::CamMatDampingSingle *camMat_damp = nullptr)
        {
            return (new ceres::AutoDiffCostFunction<
                    BAFixedDepthCostFunction<CameraModel>, 2, 4, 3, 2, 3,
                    CameraModel::kNumParams>(
                new BAFixedDepthCostFunction(depth, point2D, distortion_damping, camMat_damp)));
        }

        template <typename T>
        bool operator()(const T *const qvec, const T *const tvec,
                        const T *const d_scale_add,
                        const T *const point3D, const T *const camera_params,
                        T *residuals) const
        {
            //Calculate 3D point at fixed depth
            T pt_3d_fd[3];
            pt_3d_fd[2] = d_scale_add[0] * T(depth_) + d_scale_add[1];
            pt_3d_fd[0] = pt_3d_fd[2] * point3D[0] / point3D[2];
            pt_3d_fd[1] = pt_3d_fd[2] * point3D[1] / point3D[2];

            // Rotate and translate.
            T projection[3];
            ceres::UnitQuaternionRotatePoint(qvec, pt_3d_fd, projection);
            projection[0] += tvec[0];
            projection[1] += tvec[1];
            projection[2] += tvec[2];

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // Distort and transform to pixel space.
            CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                      &residuals[0], &residuals[1]);

            // Re-projection error.
            residuals[0] -= T(observed_x_);
            residuals[1] -= T(observed_y_);

            if (dist_damp)
            {
                CameraModel::AddDistortionDamping(camera_params, &residuals[0], &residuals[1]);
            }

            if (cam_damp)
            {
                if (cam_damp->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params, &(cam_damp->focalMid), &(cam_damp->focalRange2), &(cam_damp->f_init), &(cam_damp->fChangeMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params, &(cam_damp->fxfyRatioDiffMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params, &(cam_damp->cx_init), &(cam_damp->cy_init), &(cam_damp->imgDiag_2), &(cam_damp->cxcyDistMidRatioMax), &residuals[0], &residuals[1]);
                }
            }

            return true;
        }

    private:
        const double depth_;
        const double observed_x_;
        const double observed_y_;
        const bool dist_damp;
        const colmap::CamMatDampingSingle *cam_damp;
    };

    // Bundle adjustment cost function using fixed 3D point depths (all depth values can change linearly): variable
    // camera calibration, point parameters, global depth scale & additive term,
    // and fixed camera pose.
    template <typename CameraModel>
    class BAFixedDepthConstCamsCostFunction
    {
    public:
        BAFixedDepthConstCamsCostFunction(const Eigen::Vector4d &qvec,
                                          const Eigen::Vector3d &tvec,
                                          const double &depth,
                                          const Eigen::Vector2d &point2D,
                                          const bool distortion_damping = false,
                                          const colmap::CamMatDampingSingle *camMat_damp = nullptr)
            : qw_(qvec(0)),
              qx_(qvec(1)),
              qy_(qvec(2)),
              qz_(qvec(3)),
              tx_(tvec(0)),
              ty_(tvec(1)),
              tz_(tvec(2)),
              depth_(depth),
              observed_x_(point2D(0)),
              observed_y_(point2D(1)),
              dist_damp(distortion_damping),
              cam_damp(camMat_damp) {}

        static ceres::CostFunction *Create(const Eigen::Vector4d &qvec,
                                           const Eigen::Vector3d &tvec,
                                           const double &depth,
                                           const Eigen::Vector2d &point2D,
                                           const bool distortion_damping = false,
                                           const colmap::CamMatDampingSingle *camMat_damp = nullptr)
        {
            return (new ceres::AutoDiffCostFunction<
                    BAFixedDepthConstCamsCostFunction<CameraModel>, 2, 3, 2,
                    CameraModel::kNumParams>(
                new BAFixedDepthConstCamsCostFunction(qvec, tvec, depth, point2D, distortion_damping, camMat_damp)));
        }

        template <typename T>
        bool operator()(const T *const point3D, const T *const d_scale_add,
                        const T *const camera_params,
                        T *residuals) const
        {
            //Calculate 3D point at fixed depth
            T pt_3d_fd[3];
            pt_3d_fd[2] = d_scale_add[0] * T(depth_) + d_scale_add[1];
            pt_3d_fd[0] = pt_3d_fd[2] * point3D[0] / point3D[2];
            pt_3d_fd[1] = pt_3d_fd[2] * point3D[1] / point3D[2];

            const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

            // Rotate and translate.
            T projection[3];
            ceres::UnitQuaternionRotatePoint(qvec, pt_3d_fd, projection);
            projection[0] += T(tx_);
            projection[1] += T(ty_);
            projection[2] += T(tz_);

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // Distort and transform to pixel space.
            CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                      &residuals[0], &residuals[1]);

            // Re-projection error.
            residuals[0] -= T(observed_x_);
            residuals[1] -= T(observed_y_);

            if (dist_damp)
            {
                CameraModel::AddDistortionDamping(camera_params, &residuals[0], &residuals[1]);
            }

            if (cam_damp)
            {
                if (cam_damp->useDampingF())
                {
                    CameraModel::AddFocalDamping(camera_params, &(cam_damp->focalMid), &(cam_damp->focalRange2), &(cam_damp->f_init), &(cam_damp->fChangeMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp->useDampingFxFy())
                {
                    CameraModel::AddFxFyDamping(camera_params, &(cam_damp->fxfyRatioDiffMax), &residuals[0], &residuals[1]);
                }
                if (cam_damp->useDampingCxCy())
                {
                    CameraModel::AddCxCyDamping(camera_params, &(cam_damp->cx_init), &(cam_damp->cy_init), &(cam_damp->imgDiag_2), &(cam_damp->cxcyDistMidRatioMax), &residuals[0], &residuals[1]);
                }
            }

            return true;
        }

    private:
        const double qw_;
        const double qx_;
        const double qy_;
        const double qz_;
        const double tx_;
        const double ty_;
        const double tz_;
        const double depth_;
        const double observed_x_;
        const double observed_y_;
        const bool dist_damp;
        const colmap::CamMatDampingSingle *cam_damp;
    };

    struct BundleAdjustmentOptions
    {
        // Loss function types: Trivial (non-robust) and Cauchy (robust) loss.
        enum class LossFunctionType
        {
            TRIVIAL,
            SOFT_L1,
            CAUCHY,
            HUBER
        };
        LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

        // Scaling factor determines residual at which robustification takes place.
        double loss_function_scale = 1.0;

        // Whether to refine the focal length parameter group.
        bool refine_focal_length = true;

        // Whether to refine the principal point parameter group.
        bool refine_principal_point = false;

        // Whether to refine the extra parameter group.
        bool refine_extra_params = true;

        // Whether to refine the extrinsic parameter group.
        bool refine_extrinsics = true;

        // Whether to print a final summary.
        bool print_summary = true;

        //Use the trust region algorithm and linear solver specified within each individual BA method
        bool useDefaultSolver = true;

        int CeresCPUcnt = -1;

        // Minimum number of residuals to enable multi-threading. Note that
        // single-threaded is typically better for small bundle adjustment problems
        // due to the overhead of threading.
        //    int min_num_residuals_for_multi_threading = 50000;

        // Ceres-Solver options.
        ceres::Solver::Options solver_options;

        BundleAdjustmentOptions()
        {            
            solver_options.function_tolerance = 1e-6;
            solver_options.gradient_tolerance = 1e-10;
            solver_options.parameter_tolerance = 1e-8;
            solver_options.minimizer_progress_to_stdout = false;
            solver_options.max_num_iterations = 100;
            solver_options.max_linear_solver_iterations = 200;
            solver_options.max_solver_time_in_seconds = 2e4;
            solver_options.max_num_consecutive_invalid_steps = 10;
            solver_options.max_consecutive_nonmonotonic_steps = 10;
            solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
            solver_options.num_linear_solver_threads = -1;
#endif // CERES_VERSION_MAJOR
        }

        // Create a new loss function based on the specified options. The caller
        // takes ownership of the loss function.
        ceres::LossFunction *CreateLossFunction() const;

        bool Check() const;
    };

    class StereoBundleAdjuster
    {
    public:
        explicit StereoBundleAdjuster(BundleAdjustmentOptions options, const bool use_ceres_ownership = true);
        ~StereoBundleAdjuster()
        {            
            delete loss_function;
            if (!use_ceres_ownership_){
                for (auto &cf : cost_func_ptrs){
                    delete cf;
                }
                for (auto &lp : local_subset_para_ptrs)
                {
                    delete lp;
                }
            }
            // problem_.reset();
        }

        bool Solve(StereoBAData *corrImgs);

    private:
        void SetUp(StereoBAData *corrImgs);

        void AddImagePairToProblem(StereoBAData *corrImgs);

        void ParameterizeCameraPair(StereoBAData *corrImgs);

        void setFixedIntrinsics(colmap::Camera &cam);

        void setFixed3Dpoints(StereoBAData *corrImgs);

        const double max_reproj_error = 10.0;

        const BundleAdjustmentOptions options_;
        const bool use_ceres_ownership_;
        std::unique_ptr<ceres::Problem> problem_;
        ceres::LossFunction *loss_function = nullptr;
        ceres::Solver::Summary summary_;

        // The Quaternions added to the problem, used to set the local
        // parameterization once after setting up the problem.
        std::unordered_set<double *> parameterized_qvec_data_;
        std::vector<bool> q_used;
        std::vector<ceres::CostFunction *> cost_func_ptrs;
        std::vector<ceres::Manifold *> local_subset_para_ptrs;
    };

    class GlobalBundleAdjuster
    {
    public:
        explicit GlobalBundleAdjuster(BundleAdjustmentOptions options, const bool use_ceres_ownership = true, const bool allImgsNeeded = true, const bool allCamsNeeded = true);
        ~GlobalBundleAdjuster()
        {            
            delete loss_function;
            if (!use_ceres_ownership_)
            {
                for (auto &cf : cost_func_ptrs)
                {
                    delete cf;
                }
                for (auto &lp : local_subset_para_ptrs)
                {
                    delete lp;
                }
            }
            // problem_.reset();
        }

        bool Solve(GlobalBAData *corrImgs, std::string *dbg_info = nullptr);

    private:
        bool SetUp(GlobalBAData *corrImgs, std::string *dbg_info = nullptr);

        void AddImagesToProblem(GlobalBAData *corrImgs, std::string *dbg_info = nullptr);

        void ParameterizeCameras(GlobalBAData *corrImgs);

        void setFixedIntrinsics(colmap::Camera &cam);

        void setFixed3Dpoints(GlobalBAData *corrImgs);

        const double max_reproj_error = 10.0;
        const bool allImgsNeeded_;
        const bool allCamsNeeded_;
        bool refine_extrinsics = true;

        const BundleAdjustmentOptions options_;
        const bool use_ceres_ownership_;
        std::unique_ptr<ceres::Problem> problem_;
        ceres::LossFunction *loss_function = nullptr;
        ceres::Solver::Summary summary_;

        // The Quaternions added to the problem, used to set the local
        // parameterization once after setting up the problem.
        std::unordered_set<double *> parameterized_qvec_data_;
        std::vector<int> q_used;
        std::unordered_set<size_t> used_cams;
        std::vector<ceres::CostFunction *> cost_func_ptrs;
        std::vector<ceres::Manifold *> local_subset_para_ptrs;
    };

    class RelativePoseBundleAdjuster
    {
    public:
        explicit RelativePoseBundleAdjuster(BundleAdjustmentOptions options, const bool use_ceres_ownership = true, const bool allImgsNeeded = true, const bool allCamsNeeded = true);
        ~RelativePoseBundleAdjuster()
        {
            delete loss_function;
            if (!use_ceres_ownership_)
            {
                for (auto &cf : cost_func_ptrs)
                {
                    delete cf;
                }
                for (auto &lp : local_subset_homog_para_ptrs)
                {
                    delete lp;
                }
            }
            // problem_.reset();
        }

        bool Solve(RelativePoseBAData *corrImgs);

        bool getNotRefinedImgs(std::unordered_map<std::pair<int, int>, std::vector<std::vector<std::pair<int, int>>>, pair_hash, pair_EqualTo> &missing_restore_sequences);

    private:
        bool SetUp(RelativePoseBAData *corrImgs);

        void AddImagesToProblem(RelativePoseBAData *corrImgs);

        void ParameterizeCameras(RelativePoseBAData *corrImgs);

        void setFixedIntrinsics(colmap::Camera &cam);

        const double max_reproj_error = 20.0;
        const bool allImgsNeeded_;
        const bool allCamsNeeded_;
        bool refine_extrinsics = true;

        const BundleAdjustmentOptions options_;
        const bool use_ceres_ownership_;
        std::unique_ptr<ceres::Problem> problem_;
        ceres::LossFunction *loss_function = nullptr;
        ceres::Solver::Summary summary_;

        // The Quaternions added to the problem, used to set the local
        // parameterization once after setting up the problem.
        std::unordered_set<double *> parameterized_qvec_data_;
        std::unordered_set<double *> parameterized_tvec_data_;
        std::unordered_set<int> used_cams;
        std::unordered_set<std::pair<int, int>, pair_hash, pair_EqualTo> missed_img_combs, available_img_combs;
        std::unordered_map<std::pair<int, int>, std::vector<std::vector<std::pair<int, int>>>, pair_hash, pair_EqualTo> track_restore_missing;
        std::vector<ceres::CostFunction *> cost_func_ptrs;
        std::vector<ceres::Manifold *> local_subset_homog_para_ptrs;
    };

    class FixedDepthBundleAdjuster
    {
    public:
        explicit FixedDepthBundleAdjuster(BundleAdjustmentOptions options, const bool use_ceres_ownership = true, const bool allImgsNeeded = true, const bool allCamsNeeded = true);
        ~FixedDepthBundleAdjuster()
        {
            delete loss_function;
            if (!use_ceres_ownership_)
            {
                for (auto &cf : cost_func_ptrs)
                {
                    delete cf;
                }
                for (auto &lp : local_subset_para_ptrs)
                {
                    delete lp;
                }
            }
            // problem_.reset();
        }

        bool Solve(FixedDepthBAData *corrImgs);

    private:
        bool SetUp(FixedDepthBAData *corrImgs);

        void AddImagesToProblem(FixedDepthBAData *corrImgs);

        void ParameterizeCameras(FixedDepthBAData *corrImgs);

        void setFixedIntrinsics(colmap::Camera &cam);

        const double max_reproj_error = 50.0;
        const bool allImgsNeeded_;
        const bool allCamsNeeded_;
        bool refine_extrinsics = true;

        const BundleAdjustmentOptions options_;
        const bool use_ceres_ownership_;
        std::unique_ptr<ceres::Problem> problem_;
        ceres::LossFunction *loss_function = nullptr;
        ceres::Solver::Summary summary_;

        // The Quaternions added to the problem, used to set the local
        // parameterization once after setting up the problem.
        std::unordered_set<double *> parameterized_qvec_data_;
        std::vector<int> q_used;
        std::unordered_set<size_t> used_cams;
        std::vector<ceres::CostFunction *> cost_func_ptrs;
        std::vector<ceres::Manifold *> local_subset_para_ptrs;
    };
}
