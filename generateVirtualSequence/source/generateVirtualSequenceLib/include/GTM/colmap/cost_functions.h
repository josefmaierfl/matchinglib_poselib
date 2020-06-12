// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_BASE_COST_FUNCTIONS_H_
#define COLMAP_SRC_BASE_COST_FUNCTIONS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

    template<typename SCALAR, int N>
    int get_integer_part( const ceres::Jet<SCALAR, N>& x );
    int get_integer_part( double x );

// Standard bundle adjustment cost function for variable
// camera pose and calibration and point parameters.
template <typename CameraModel>
class BundleAdjustmentCostFunction {
 public:
  explicit BundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 3,
            CameraModel::kNumParams>(
        new BundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
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

    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class BundleAdjustmentConstantPoseCostFunction {
 public:
  BundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec,
                                           const Eigen::Vector3d& tvec,
                                           const Eigen::Vector2d& point2D)
      : qw_(qvec(0)),
        qx_(qvec(1)),
        qy_(qvec(2)),
        qz_(qvec(3)),
        tx_(tvec(0)),
        ty_(tvec(1)),
        tz_(tvec(2)),
        observed_x_(point2D(0)),
        observed_y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec,
                                     const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentConstantPoseCostFunction<CameraModel>, 2, 3,
            CameraModel::kNumParams>(
        new BundleAdjustmentConstantPoseCostFunction(qvec, tvec, point2D)));
  }

  template <typename T>
  bool operator()(const T* const point3D, const T* const camera_params,
                  T* residuals) const {
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
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel>
class RigBundleAdjustmentCostFunction {
 public:
  explicit RigBundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            RigBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 4, 3, 3,
            CameraModel::kNumParams>(
        new RigBundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const rig_qvec, const T* const rig_tvec,
                  const T* const rel_qvec, const T* const rel_tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
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

    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `HomogeneousVectorParameterization`.
class RelativePoseCostFunction {
 public:
  RelativePoseCostFunction(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
      : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& x1,
                                     const Eigen::Vector2d& x2) {
    return (new ceres::AutoDiffCostFunction<RelativePoseCostFunction, 1, 4, 3>(
        new RelativePoseCostFunction(x1, x2)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  T* residuals) const {
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

// Cost function for refining two-view geometry based on fixed depth maps of image 1 and image 2.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be rotated and translated by a relative pose including scaling
    template <typename CameraModel>
    class RelativePoseFixedDepthCostFunction {
    public:
        explicit RelativePoseFixedDepthCostFunction(const Eigen::Vector2i& x1, const double &depth1, const Eigen::Ref<const Eigen::MatrixXd>& depthMap2)
                : x1_(x1(0)), y1_(x1(1)), depth1_(depth1), depthMap2_(depthMap2) {}

        static ceres::CostFunction* Create(const Eigen::Vector2i& x1,
                                           const double &depth1,
                                           const Eigen::Ref<const Eigen::MatrixXd>& depthMap2) {
            return (new ceres::AutoDiffCostFunction<RelativePoseFixedDepthCostFunction<CameraModel>, 3, 4, 3,
                    CameraModel::kNumParams, CameraModel::kNumParams>(
                    new RelativePoseFixedDepthCostFunction(x1, depth1, depthMap2)));
        }

        template <typename T>
        bool operator()(const T* const qvec, const T* const tvec, const T* const camera_params1, const T* const camera_params2,
                        T* residuals) const {
            Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
            ceres::QuaternionToRotation(qvec, R.data());
//            Eigen::Matrix<T, 3, 1, Eigen::RowMajor> t(tvec[0], tvec[1], tvec[2]);

            // Undistort and transform to world space.
            T x1w, y1w, x2i, y2i;
            CameraModel::ImageToWorld(camera_params1, T(x1_), T(y1_),
                                      &x1w, &y1w);
            Eigen::Matrix<T, 3, 1> p1(x1w, y1w, T(1.));
            p1 *= T(depth1_);
            p1 = R * p1;
            p1(0) += tvec[0];
            p1(1) += tvec[1];
            p1(2) += tvec[2];

            residuals[0] = p1(0);
            residuals[1] = p1(1);
            residuals[2] = p1(2);

            p1 /= p1(2);

            // Distort and transform to pixel space.
            CameraModel::WorldToImage(camera_params2, p1(0), p1(1),
                                      &x2i, &y2i);

            // Bilinear interpolation of depth in 2nd image
            double dtl, dtr, dbl, dbr;
            T xl = ceres::floor(x2i);
            T xr = ceres::ceil(ceres::abs(x2i));
            T yt = ceres::floor(y2i);
            T yb = ceres::ceil(ceres::abs(y2i));
            int xli = get_integer_part(xl);
            int xri = get_integer_part(xr);
            int yti = get_integer_part(yt);
            int ybi = get_integer_part(yb);
            if((xli < 0) || (xri >= depthMap2_.cols()) || (yti < 0) || (ybi >= depthMap2_.rows())){
                bool skip = false;
                if(xli == -1){
                    xli = xri = 0;
                }else if(xri == depthMap2_.cols()){
                    xli = xri = depthMap2_.cols() - 1;
                }else if((xli < 0) || (xri >= depthMap2_.cols())){
                    skip = true;
                }
                if(yti == -1){
                    yti = ybi = 0;
                }else if(ybi == depthMap2_.rows()){
                    yti = ybi = depthMap2_.rows() - 1;
                }else if((yti < 0) || (ybi >= depthMap2_.rows())){
                    skip = true;
                }
                if(skip) {
                    T x2b, y2b, x2b1, y2b1;
                    if(xli < 0){
                        x2b = T(0);
                    }else if(xri >= depthMap2_.cols()){
                        x2b = T(depthMap2_.cols() - 1);
                    }else{
                        x2b = x2i;
                    }
                    if(yti < 0){
                        y2b = T(0);
                    }else if(ybi >= depthMap2_.rows()){
                        y2b = T(depthMap2_.rows() - 1);
                    }
                    CameraModel::ImageToWorld(camera_params2, x2b, y2b,
                                              &x2b1, &y2b1);
                    Eigen::Matrix<T, 3, 1> p2(x2b1, y2b1, T(1.));
                    p2 *= T(depth1_);
                    residuals[0] -= p2(0);
                    residuals[1] -= p2(1);
                    residuals[2] -= p2(2);
                    return true;
                }
            }
            dtl = depthMap2_(yti, xli);
            dtr = depthMap2_(yti, xri);
            dbl = depthMap2_(ybi, xli);
            dbr = depthMap2_(ybi, xri);
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
                p1 *= T(0.8 * depth1_);
                residuals[0] -= p1(0);
                residuals[1] -= p1(1);
                residuals[2] -= p1(2);
                return true;
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
            T xdiff = xr - xl;
            T dt, db, d2;
            if(xdiff < T(1e-4)){
                dt = T(dtl);
                db = T(dbl);
            }else {
                T xrDiff = xr - x2i;
                T xlDiff = x2i - xl;
                dt = ((T(dtl) * xrDiff + T(dtr) * xlDiff)) / xdiff;
                db = ((T(dbl) * xrDiff + T(dbr) * xlDiff)) / xdiff;
            }
            T ydiff = yb - yt;
            if(ydiff < T(1e-4)){
                d2 = dt;
            }else {
                d2 = ((dt * (yb - y2i) + db * (y2i - yt))) / ydiff;
            }

            p1 *= d2;

            residuals[0] -= p1(0);
            residuals[1] -= p1(1);
            residuals[2] -= p1(2);

            return true;
        }

    private:
        const double x1_;
        const double y1_;
        const double depth1_;
        const Eigen::Ref<const Eigen::MatrixXd> depthMap2_;
    };

    int get_integer_part( double x )
    { return static_cast<int>(x); }

    template<typename SCALAR, int N>
    int get_integer_part( const ceres::Jet<SCALAR, N>& x )
    { return static_cast<int>(x.a); }

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_COST_FUNCTIONS_H_
