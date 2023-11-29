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

#include "ceres/ba_cost_functions.h"

#include <thread>
#include <iomanip>

using namespace colmap;

namespace poselib
{
    void PrintHeading1(const std::string &heading);
    void PrintHeading2(const std::string &heading);
    void PrintSolverSummary(const ceres::Solver::Summary &summary);
    int GetEffectiveNumThreads(const int num_threads);
    void SetDistortionLowerUpperBounds(std::unique_ptr<ceres::Problem> &problem_, colmap::Camera &cam);

    ceres::LossFunction *BundleAdjustmentOptions::CreateLossFunction() const
    {
        ceres::LossFunction *loss_function = nullptr;
        switch (loss_function_type)
        {
        case LossFunctionType::TRIVIAL:
            loss_function = new ceres::TrivialLoss();
            break;
        case LossFunctionType::SOFT_L1:
            loss_function = new ceres::SoftLOneLoss(loss_function_scale);
            break;
        case LossFunctionType::CAUCHY:
            loss_function = new ceres::CauchyLoss(loss_function_scale);
            break;
        case LossFunctionType::HUBER:
            loss_function = new ceres::HuberLoss(loss_function_scale);
            break;
        default:
            loss_function = new ceres::TrivialLoss();
            break;
        }
        CHECK_NOTNULL(loss_function);
        return loss_function;
    }

    bool BundleAdjustmentOptions::Check() const
    {
        if (loss_function_scale < 0)
        {
            return false;
        }
        return true;
    }

    StereoBundleAdjuster::StereoBundleAdjuster(BundleAdjustmentOptions options, const bool use_ceres_ownership)
        : options_(std::move(options)), use_ceres_ownership_(use_ceres_ownership)
    {
        CHECK(options_.Check());
        loss_function = options_.CreateLossFunction();
    }

    bool StereoBundleAdjuster::Solve(StereoBAData *corrImgs)
    {
        ceres::Problem::Options problem_opts;
        if (use_ceres_ownership_){
            problem_opts.enable_fast_removal = true;
            problem_opts.cost_function_ownership = ceres::TAKE_OWNERSHIP;
            problem_opts.disable_all_safety_checks = false;
            problem_opts.manifold_ownership = ceres::TAKE_OWNERSHIP;
        }else{
            problem_opts.enable_fast_removal = false;
            problem_opts.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_opts.disable_all_safety_checks = false;
            problem_opts.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        }
        problem_opts.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        if (problem_){
            problem_.reset(new ceres::Problem(problem_opts));
        }else{
            problem_ = std::make_unique<ceres::Problem>(problem_opts);
        }
        parameterized_qvec_data_.clear();

        //    ceres::LossFunction* loss_function = options_.CreateLossFunction();
        SetUp(corrImgs);

        if (problem_->NumResiduals() < 20)
        {
            return false;
        }

        ceres::Solver::Options solver_options = options_.solver_options;
        if(options_.useDefaultSolver){
            solver_options.linear_solver_type = ceres::DENSE_SCHUR;
        }

        int num_threads = GetEffectiveNumThreads(solver_options.num_threads);
        if ((options_.CeresCPUcnt > 0) && (num_threads > options_.CeresCPUcnt))
        {
            num_threads = options_.CeresCPUcnt;
        }
        solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
        num_threads = GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
        if ((options_.CeresCPUcnt > 0) && (num_threads > options_.CeresCPUcnt))
        {
            num_threads = options_.CeresCPUcnt;
        }
        solver_options.num_linear_solver_threads = num_threads;
#endif // CERES_VERSION_MAJOR

        std::string solver_error;
        CHECK(solver_options.IsValid(&solver_error)); // << solver_error;

        ceres::Solve(solver_options, problem_.get(), &summary_);

        if (solver_options.minimizer_progress_to_stdout)
        {
            std::cout << std::endl;
        }

        if (options_.print_summary)
        {
            PrintHeading2("Stereo bundle adjustment report");
            PrintSolverSummary(summary_);
        }

        return summary_.IsSolutionUsable();
    }

    void StereoBundleAdjuster::SetUp(StereoBAData *corrImgs)
    {
        AddImagePairToProblem(corrImgs);
        if (parameterized_qvec_data_.empty()){
            return;
        }
        ParameterizeCameraPair(corrImgs);
        setFixed3Dpoints(corrImgs);
    }

    void StereoBundleAdjuster::AddImagePairToProblem(StereoBAData *corrImgs)
    {
        double max_squared_reproj_error = max_reproj_error * max_reproj_error;

        colmap::Camera &camera1 = corrImgs->cam1;
        colmap::Camera &camera2 = corrImgs->cam2;

        double *qvec_data = nullptr;
        double *tvec_data = nullptr;
        double *camera_params_data1 = camera1.ParamsData();
        double *camera_params_data2 = camera2.ParamsData();

        if(corrImgs->inCamCoordinates){
            const double fm = (camera1.MeanFocalLength() + camera2.MeanFocalLength()) / 2.0;
            max_squared_reproj_error /= fm * fm;
        }

        // CostFunction assumes unit quaternions.
        corrImgs->quat_rel = colmap::NormalizeQuaternion(corrImgs->quat_rel);
        qvec_data = corrImgs->quat_rel.data();
        tvec_data = corrImgs->t_rel.data();

        // The number of added observations for the current image.
        size_t num_observations = 0;

        if (!corrImgs->points3d.empty()){
            q_used = std::vector<bool>(corrImgs->points3d.size(), true);
        }

        // Add residuals to bundle adjustment problem.
        size_t idx = 0;
        for (const auto &corr : corrImgs->corrs)
        {
            if (!corrImgs->points3d.empty() && corrImgs->calcReprojectionError(corr.first, corr.second, corrImgs->points3d[idx]) > max_squared_reproj_error)
            {
                q_used[idx] = false;
                idx++;
                continue;
            }

            num_observations += 1;

            double *point3d_data = nullptr;
            if (!corrImgs->points3d.empty()){
                point3d_data = corrImgs->points3d[idx++].data();
            }

            ceres::CostFunction *cost_function = nullptr;
            if(corrImgs->inCamCoordinates && corrImgs->points3d.empty())
            {
                cost_function = RelativePoseCostFunction::Create(corr.first, corr.second);
                problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
            }
            else if(corrImgs->inCamCoordinates)
            {
                cost_function = StereoBundleAdjustmentCostFunctionCamCoordinates::Create(corr.first, corr.second);
                problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                           tvec_data, point3d_data);
            }
            else
            {
                switch (camera1.ModelId())
                {
#define CAMERA_MODEL_CASE(CameraModel)                                                                                                    \
    case CameraModel::kModelId:                                                                                                           \
        cost_function =                                                                                                                   \
            StereoBundleAdjustmentCostFunction<CameraModel>::Create(corr.first,                                                           \
                                                                    corr.second,                                                          \
                                                                    corrImgs->dist_damp,                                                  \
                                                                    corrImgs->camMat_damp ? &(corrImgs->camMatDampPars.at(0)) : nullptr,  \
                                                                    corrImgs->camMat_damp ? &(corrImgs->camMatDampPars.at(1)) : nullptr); \
                                                                                                                                          \
        break;

                    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                }
                problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                        tvec_data, point3d_data, camera_params_data1, camera_params_data2);
            }
            cost_func_ptrs.push_back(cost_function);
        }

        if (num_observations > 0)
        {
            parameterized_qvec_data_.insert(qvec_data);
        }
    }

    void StereoBundleAdjuster::ParameterizeCameraPair(StereoBAData *corrImgs)
    {
        const bool constant_camera = !options_.refine_focal_length &&
                                     !options_.refine_principal_point &&
                                     !options_.refine_extra_params;
        for (double *qvec_data : parameterized_qvec_data_)
        {
            ceres::Manifold *quaternion_parameterization =
                new ceres::QuaternionManifold;
            local_subset_para_ptrs.push_back(quaternion_parameterization);
            problem_->SetManifold(qvec_data, quaternion_parameterization);
        }

        if (!corrImgs->inCamCoordinates)
        {
            if (constant_camera)
            {
                problem_->SetParameterBlockConstant(corrImgs->cam1.ParamsData());
                problem_->SetParameterBlockConstant(corrImgs->cam2.ParamsData());
            }
            else
            {
                setFixedIntrinsics(corrImgs->cam1);
                setFixedIntrinsics(corrImgs->cam2);
            }
        }
    }

    void StereoBundleAdjuster::setFixedIntrinsics(colmap::Camera &cam)
    {
        std::vector<int> const_camera_params;

        if (!options_.refine_focal_length)
        {
            const std::vector<size_t> &params_idxs = cam.FocalLengthIdxs();
            const_camera_params.insert(const_camera_params.end(),
                                       params_idxs.begin(), params_idxs.end());
        }
        if (!options_.refine_principal_point)
        {
            const std::vector<size_t> &params_idxs = cam.PrincipalPointIdxs();
            const_camera_params.insert(const_camera_params.end(),
                                       params_idxs.begin(), params_idxs.end());
        }
        if (!options_.refine_extra_params)
        {
            const std::vector<size_t> &params_idxs = cam.ExtraParamsIdxs();
            if (!params_idxs.empty()){
                const_camera_params.insert(const_camera_params.end(),
                                           params_idxs.begin(), params_idxs.end());
            }
        }
        // else
        // {
        //     SetDistortionLowerUpperBounds(problem_, cam);
        // }

        if (const_camera_params.size() > 0)
        {
            ceres::Manifold *camera_params_parameterization =
                new ceres::SubsetManifold(
                    static_cast<int>(cam.NumParams()), const_camera_params);
            local_subset_para_ptrs.push_back(camera_params_parameterization);
            problem_->SetManifold(cam.ParamsData(),
                                          camera_params_parameterization);
        }
    }

    void StereoBundleAdjuster::setFixed3Dpoints(StereoBAData *corrImgs){
        if(corrImgs->points3d.empty() || corrImgs->constant_points3d.empty()){
            return;
        }
        for (const auto &point3D_id : corrImgs->constant_points3d)
        {
            if (q_used[point3D_id]){
                Eigen::Vector3d &point3D = corrImgs->points3d.at(point3D_id);
                problem_->SetParameterBlockConstant(point3D.data());
            }
        }
    }

    GlobalBundleAdjuster::GlobalBundleAdjuster(BundleAdjustmentOptions options, const bool use_ceres_ownership, const bool allImgsNeeded, const bool allCamsNeeded)
        : allImgsNeeded_(allImgsNeeded), allCamsNeeded_(allCamsNeeded), options_(std::move(options)), use_ceres_ownership_(use_ceres_ownership)
    {
        CHECK(options_.Check());
        loss_function = options_.CreateLossFunction();
    }

    bool GlobalBundleAdjuster::Solve(GlobalBAData *corrImgs, std::string *dbg_info)
    {
        if (!corrImgs->Check()){
            return false;
        }
        ceres::Problem::Options problem_opts;
        if (use_ceres_ownership_)
        {
            problem_opts.enable_fast_removal = true;
            problem_opts.cost_function_ownership = ceres::TAKE_OWNERSHIP;
            problem_opts.disable_all_safety_checks = false;
            problem_opts.manifold_ownership = ceres::TAKE_OWNERSHIP;
        }
        else
        {
            problem_opts.enable_fast_removal = false;
            problem_opts.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_opts.disable_all_safety_checks = false;
            problem_opts.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        }
        problem_opts.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        if (problem_)
        {
            problem_.reset(new ceres::Problem(problem_opts));
        }
        else
        {
            problem_ = std::make_unique<ceres::Problem>(problem_opts);
        }
        parameterized_qvec_data_.clear();

        if (!SetUp(corrImgs, dbg_info))
        {
            std::cerr << "Setting up BA problem failed" << std::endl;
            return false;
        }

        if (problem_->NumResiduals() < 20)
        {
            std::cerr << "Too less residuals for BA remaining" << std::endl;
            return false;
        }        

        ceres::Solver::Options solver_options = options_.solver_options;
        if (options_.useDefaultSolver){
            solver_options.linear_solver_type = ceres::DENSE_SCHUR; // ceres::SPARSE_NORMAL_CHOLESKY
            if (options_.refine_extra_params)
            {
                solver_options.linear_solver_type = ceres::DENSE_SCHUR;//ceres::SPARSE_NORMAL_CHOLESKY;
                solver_options.trust_region_strategy_type = ceres::DOGLEG; //ceres::LEVENBERG_MARQUARDT
                solver_options.dogleg_type = ceres::SUBSPACE_DOGLEG;       //ceres::TRADITIONAL_DOGLEG
                solver_options.use_nonmonotonic_steps = true;
                solver_options.max_consecutive_nonmonotonic_steps = 6; //5
                solver_options.min_trust_region_radius = 1e-16;        //1e-32
            }
        }

        int num_threads = GetEffectiveNumThreads(solver_options.num_threads);
        if ((options_.CeresCPUcnt > 0) && (num_threads > options_.CeresCPUcnt))
        {
            num_threads = options_.CeresCPUcnt;
        }
        solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
        num_threads = GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
        if ((options_.CeresCPUcnt > 0) && (num_threads > options_.CeresCPUcnt))
        {
            num_threads = options_.CeresCPUcnt;
        }
        solver_options.num_linear_solver_threads = num_threads;
#endif // CERES_VERSION_MAJOR

        std::string solver_error;
        CHECK(solver_options.IsValid(&solver_error)); // << solver_error;

        ceres::Solve(solver_options, problem_.get(), &summary_);

        if (solver_options.minimizer_progress_to_stdout)
        {
            std::cout << std::endl;
        }

        if (options_.print_summary)
        {
            PrintHeading2("Global bundle adjustment report");
            PrintSolverSummary(summary_);
        }

        return summary_.IsSolutionUsable();
    }

    bool GlobalBundleAdjuster::SetUp(GlobalBAData *corrImgs, std::string *dbg_info)
    {
        AddImagesToProblem(corrImgs, dbg_info);
        if (refine_extrinsics && parameterized_qvec_data_.empty())
        {
            return false;
        }
        if(allCamsNeeded_ && used_cams.size() != corrImgs->cams.size()){
            return false;
        }
        ParameterizeCameras(corrImgs);
        setFixed3Dpoints(corrImgs);
        return true;
    }

    void GlobalBundleAdjuster::AddImagesToProblem(GlobalBAData *corrImgs, std::string *dbg_info)
    {
        double max_squared_reproj_error = max_reproj_error * max_reproj_error;
        const size_t nr_imgs = corrImgs->imgs_to_cam_idx.size();
        q_used = std::vector<int>(corrImgs->points3d.size(), 1);
        for (size_t i = 0; i < nr_imgs; ++i){
            const size_t cam_nr = corrImgs->imgs_to_cam_idx.at(i);
            colmap::Camera &cam = corrImgs->cams[cam_nr];

            double *qvec_data = nullptr;
            double *tvec_data = nullptr;
            double *camera_params_data = cam.ParamsData();

            Eigen::Vector4d &quat = corrImgs->quats[i];
            Eigen::Vector3d &t = corrImgs->ts[i];

            // CostFunction assumes unit quaternions.
            quat = colmap::NormalizeQuaternion(quat);
            qvec_data = quat.data();
            tvec_data = t.data();

            const colmap::CamMatDampingSingle *camMatDampPars = corrImgs->camMat_damp ? &(corrImgs->camMatDampPars.at(cam_nr)) : nullptr;

            bool pose_constant = !options_.refine_extrinsics;
            if (!corrImgs->constant_poses.empty())
            {
                pose_constant |= corrImgs->constant_poses.at(i);
            }

            // The number of added observations for the current image.
            size_t num_observations = 0;

            // Add residuals to bundle adjustment problem.
            const std::vector<Eigen::Vector2d> &corrs_this = corrImgs->corrs.at(i);
            const std::vector<size_t> &pt_3d_idx = corrImgs->points3d_idx.at(i);
            size_t idx = 0;
            for (const auto &corr : corrs_this)
            {
                const size_t &ptx_3d_idx = pt_3d_idx.at(idx++);
                Eigen::Vector3d &pt_3d = corrImgs->points3d[ptx_3d_idx];
                if (corrImgs->calcReprojectionError(corr, pt_3d, quat, t, cam) > max_squared_reproj_error)
                {
                    q_used[ptx_3d_idx]--;
                    continue;
                }
                q_used[ptx_3d_idx] = static_cast<int>(nr_imgs);

                num_observations += 1;

                double *point3d_data = pt_3d.data();
                ceres::CostFunction *cost_function = nullptr;
                if (pose_constant)
                {
                    switch (cam.ModelId())
                    {
#define CAMERA_MODEL_CASE(CameraModel)                                                         \
    case CameraModel::kModelId:                                                                \
        cost_function =                                                                        \
            BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(quat,                \
                                                                          t,                   \
                                                                          corr,                \
                                                                          corrImgs->dist_damp, \
                                                                          camMatDampPars);     \
                                                                                               \
        break;

                        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, point3d_data, camera_params_data);
                }
                else
                {
                    switch (cam.ModelId())
                    {
#define CAMERA_MODEL_CASE(CameraModel)                                             \
    case CameraModel::kModelId:                                                    \
        cost_function =                                                            \
            BundleAdjustmentCostFunction<CameraModel>::Create(corr,                \
                                                              corrImgs->dist_damp, \
                                                              camMatDampPars);     \
                                                                                   \
        break;

                        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, point3d_data, camera_params_data);
                }
                cost_func_ptrs.push_back(cost_function);
            }

            if (!pose_constant && num_observations > 0)
            {
                parameterized_qvec_data_.insert(qvec_data);                
            }
            else if (num_observations == 0 && allImgsNeeded_){
                parameterized_qvec_data_.clear();
                return;
            }
            if (num_observations > 0){
                used_cams.emplace(cam_nr);
            }
            if (parameterized_qvec_data_.empty() && !used_cams.empty())
            {
                refine_extrinsics = false;
            }
        }
    }

    void GlobalBundleAdjuster::ParameterizeCameras(GlobalBAData *corrImgs)
    {
        const bool constant_camera = !options_.refine_focal_length &&
                                     !options_.refine_principal_point &&
                                     !options_.refine_extra_params;        
        for (double *qvec_data : parameterized_qvec_data_)
        {
            ceres::Manifold *quaternion_parameterization =
                new ceres::QuaternionManifold;
            local_subset_para_ptrs.push_back(quaternion_parameterization);
            problem_->SetManifold(qvec_data, quaternion_parameterization);
        }

        if (constant_camera)
        {
            for(auto &cam : corrImgs->cams){
                problem_->SetParameterBlockConstant(cam.ParamsData());
            }
        }
        else
        {
            std::vector<bool> constant_cams;
            const size_t nr_cams = corrImgs->cams.size();
            if (corrImgs->constant_cams.empty())
            {
                constant_cams = std::vector<bool>(nr_cams, false);
            }
            else
            {
                constant_cams = corrImgs->constant_cams;
            }
            for (size_t i = 0; i < nr_cams; ++i){
                colmap::Camera &cam = corrImgs->cams[i];
                if (constant_cams.at(i))
                {
                    problem_->SetParameterBlockConstant(cam.ParamsData());
                }
                else
                {
                    setFixedIntrinsics(cam);
                }
            }
        }
    }

    void GlobalBundleAdjuster::setFixedIntrinsics(colmap::Camera &cam)
    {
        std::vector<int> const_camera_params;

        if (!options_.refine_focal_length)
        {
            const std::vector<size_t> &params_idxs = cam.FocalLengthIdxs();
            const_camera_params.insert(const_camera_params.end(),
                                       params_idxs.begin(), params_idxs.end());
        }
        if (!options_.refine_principal_point)
        {
            const std::vector<size_t> &params_idxs = cam.PrincipalPointIdxs();
            const_camera_params.insert(const_camera_params.end(),
                                       params_idxs.begin(), params_idxs.end());
        }
        if (!options_.refine_extra_params)
        {
            const std::vector<size_t> &params_idxs = cam.ExtraParamsIdxs();
            if (!params_idxs.empty())
            {
                const_camera_params.insert(const_camera_params.end(),
                                           params_idxs.begin(), params_idxs.end());
            }
        }
        // else
        // {
        //     SetDistortionLowerUpperBounds(problem_, cam);
        // }

        if (const_camera_params.size() > 0)
        {
            ceres::Manifold *camera_params_parameterization =
                new ceres::SubsetManifold(
                    static_cast<int>(cam.NumParams()), const_camera_params);
            local_subset_para_ptrs.push_back(camera_params_parameterization);
            problem_->SetManifold(cam.ParamsData(),
                                          camera_params_parameterization);
        }
    }

    void GlobalBundleAdjuster::setFixed3Dpoints(GlobalBAData *corrImgs)
    {
        if (corrImgs->constant_points3d.empty())
        {
            return;
        }
        for (const auto &point3D_id : corrImgs->constant_points3d)
        {
            if (q_used[point3D_id] > 0){
                Eigen::Vector3d &point3D = corrImgs->points3d.at(point3D_id);
                problem_->SetParameterBlockConstant(point3D.data());
            }
        }
    }

    RelativePoseBundleAdjuster::RelativePoseBundleAdjuster(BundleAdjustmentOptions options, const bool use_ceres_ownership, const bool allImgsNeeded, const bool allCamsNeeded)
        : allImgsNeeded_(allImgsNeeded), allCamsNeeded_(allCamsNeeded), options_(std::move(options)), use_ceres_ownership_(use_ceres_ownership)
    {
        CHECK(options_.Check());
        loss_function = options_.CreateLossFunction();
    }

    bool RelativePoseBundleAdjuster::Solve(RelativePoseBAData *corrImgs)
    {
        if (!corrImgs->Check())
        {
            return false;
        }
        ceres::Problem::Options problem_opts;
        if (use_ceres_ownership_)
        {
            problem_opts.enable_fast_removal = true;
            problem_opts.cost_function_ownership = ceres::TAKE_OWNERSHIP;
            problem_opts.disable_all_safety_checks = false;
            problem_opts.manifold_ownership = ceres::TAKE_OWNERSHIP;
        }
        else
        {
            problem_opts.enable_fast_removal = false;
            problem_opts.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_opts.disable_all_safety_checks = false;
            problem_opts.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        }
        problem_opts.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        if (problem_)
        {
            problem_.reset(new ceres::Problem(problem_opts));
        }
        else
        {
            problem_ = std::make_unique<ceres::Problem>(problem_opts);
        }
        parameterized_qvec_data_.clear();

        if (!SetUp(corrImgs))
        {
            std::cerr << "Setting up BA problem failed" << std::endl;
            return false;
        }

        if (problem_->NumResiduals() < 20)
        {
            std::cerr << "Too less residuals for BA remaining" << std::endl;
            return false;
        }

        ceres::Solver::Options solver_options = options_.solver_options;
        if (options_.useDefaultSolver){
            solver_options.linear_solver_type = ceres::DENSE_SCHUR;
            // if (options_.refine_extra_params)
            // {
                // solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                solver_options.trust_region_strategy_type = ceres::DOGLEG; //ceres::LEVENBERG_MARQUARDT
                solver_options.dogleg_type = ceres::SUBSPACE_DOGLEG;       //ceres::TRADITIONAL_DOGLEG
                solver_options.use_nonmonotonic_steps = true;
                solver_options.max_consecutive_nonmonotonic_steps = 6; //5
                solver_options.min_trust_region_radius = 1e-16;        //1e-32
            // }
        }

        int num_threads = GetEffectiveNumThreads(solver_options.num_threads);
        if ((options_.CeresCPUcnt > 0) && (num_threads > options_.CeresCPUcnt))
        {
            num_threads = options_.CeresCPUcnt;
        }
        solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
        num_threads = GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
        if ((options_.CeresCPUcnt > 0) && (num_threads > options_.CeresCPUcnt))
        {
            num_threads = options_.CeresCPUcnt;
        }
        solver_options.num_linear_solver_threads = num_threads;
#endif // CERES_VERSION_MAJOR

        std::string solver_error;
        CHECK(solver_options.IsValid(&solver_error)); // << solver_error;

        ceres::Solve(solver_options, problem_.get(), &summary_);

        if (solver_options.minimizer_progress_to_stdout)
        {
            std::cout << std::endl;
        }

        if (options_.print_summary)
        {
            PrintHeading2("Relative pose bundle adjustment report");
            PrintSolverSummary(summary_);
        }

        return summary_.IsSolutionUsable();
    }

    bool RelativePoseBundleAdjuster::SetUp(RelativePoseBAData *corrImgs)
    {
        AddImagesToProblem(corrImgs);
        if (refine_extrinsics && parameterized_qvec_data_.empty())
        {
            return false;
        }
        if (allCamsNeeded_ && used_cams.size() != corrImgs->cams.size())
        {
            return false;
        }
        ParameterizeCameras(corrImgs);
        return true;
    }

    void RelativePoseBundleAdjuster::AddImagesToProblem(RelativePoseBAData *corrImgs)
    {
        double focal_sum = 0.;
        for(const auto &cx : corrImgs->cams){
            focal_sum += cx.second.MeanFocalLength();
        }
        focal_sum /= static_cast<double>(corrImgs->cams.size());
        double max_squared_reproj_error = max_reproj_error / focal_sum;
        max_squared_reproj_error *= max_squared_reproj_error;

        for (const auto &corrs : corrImgs->corrs)
        {
            const int cam_nr1 = corrs.first.first;
            const int cam_nr2 = corrs.first.second;
            colmap::Camera &cam1 = corrImgs->cams.at(cam_nr1);
            colmap::Camera &cam2 = corrImgs->cams.at(cam_nr2);

            const colmap::CamMatDampingSingle *camMatDampPars1 = corrImgs->camMat_damp ? &(corrImgs->camMatDampPars.at(cam_nr1)) : nullptr;
            const colmap::CamMatDampingSingle *camMatDampPars2 = corrImgs->camMat_damp ? &(corrImgs->camMatDampPars.at(cam_nr2)) : nullptr;

            double *qvec_data = nullptr;
            double *tvec_data = nullptr;
            double *camera_params_data1 = cam1.ParamsData();
            double *camera_params_data2 = cam2.ParamsData();

            Eigen::Vector4d &quat = corrImgs->quats.at(corrs.first);
            Eigen::Vector3d &t = corrImgs->ts.at(corrs.first);

            // CostFunction assumes unit quaternions.
            quat = colmap::NormalizeQuaternion(quat);
            qvec_data = quat.data();
            tvec_data = t.data();

            bool pose_constant = !options_.refine_extrinsics;
            if (!corrImgs->constant_poses.empty())
            {
                pose_constant |= corrImgs->constant_poses.at(corrs.first);
            }

            // The number of added observations for the current image.
            size_t num_observations = 0;

            // Add residuals to bundle adjustment problem.
            for (size_t idx = 0; idx < corrs.second.first.size(); idx++)
            {
                const Eigen::Vector2d &pt1 = corrs.second.first.at(idx);
                const Eigen::Vector2d &pt2 = corrs.second.second.at(idx);
                if (corrImgs->calcReprojectionError(pt1, pt2, quat, t, cam1, cam2) > max_squared_reproj_error)
                {
                    continue;
                }

                num_observations += 1;

                ceres::CostFunction *cost_function = nullptr;
                if (pose_constant)
                {
                    switch (cam1.ModelId())
                    {
#define CAMERA_MODEL_CASE(CameraModel)                                              \
    case CameraModel::kModelId:                                                     \
        cost_function =                                                             \
            RelativeConstPoseCostFunction<CameraModel>::Create(quat,                \
                                                               t,                   \
                                                               pt1,                 \
                                                               pt2,                 \
                                                               corrImgs->dist_damp, \
                                                               camMatDampPars1,     \
                                                               camMatDampPars2);    \
        break;

                        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, camera_params_data1, camera_params_data2);
                }
                else
                {
                    switch (cam1.ModelId())
                    {
#define CAMERA_MODEL_CASE(CameraModel)                                                  \
    case CameraModel::kModelId:                                                         \
        cost_function =                                                                 \
            RelativePoseImgCoordsCostFunction<CameraModel>::Create(pt1,                 \
                                                                   pt2,                 \
                                                                   corrImgs->dist_damp, \
                                                                   camMatDampPars1,     \
                                                                   camMatDampPars2);    \
                                                                                        \
        break;

                        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, camera_params_data1, camera_params_data2);
                }
                cost_func_ptrs.push_back(cost_function);
            }

            if (!pose_constant && num_observations > 0)
            {
                parameterized_qvec_data_.insert(qvec_data);
                parameterized_tvec_data_.insert(tvec_data);
            }
            else if (num_observations == 0 && allImgsNeeded_)
            {
                parameterized_qvec_data_.clear();
                parameterized_tvec_data_.clear();
                return;
            }
            else if (num_observations == 0 && !allImgsNeeded_)
            {
                missed_img_combs.emplace(corrs.first);
            }
            if (num_observations > 0)
            {
                used_cams.emplace(corrs.first.first);
                used_cams.emplace(corrs.first.second);
                available_img_combs.emplace(corrs.first);
            }
            if (parameterized_qvec_data_.empty() && !used_cams.empty())
            {
                refine_extrinsics = false;
            }
        }

        if (!allImgsNeeded_ && !missed_img_combs.empty())
        {
            //Check if the missing combination can be restored from the rest
            if (!getAvailablePairCombinationsForMissing(missed_img_combs, available_img_combs, track_restore_missing)){
                parameterized_qvec_data_.clear();
                parameterized_tvec_data_.clear();
                return;
            }
        }
    }

    bool RelativePoseBundleAdjuster::getNotRefinedImgs(std::unordered_map<std::pair<int, int>, std::vector<std::vector<std::pair<int, int>>>, pair_hash, pair_EqualTo> &missing_restore_sequences)
    {
        if (track_restore_missing.empty()){
            return false;
        }
        missing_restore_sequences = track_restore_missing;
        return true;
    }

    void RelativePoseBundleAdjuster::ParameterizeCameras(RelativePoseBAData *corrImgs)
    {
        const bool constant_camera = !options_.refine_focal_length &&
                                     !options_.refine_principal_point &&
                                     !options_.refine_extra_params;
        for (double *qvec_data : parameterized_qvec_data_)
        {
            ceres::Manifold *quaternion_parameterization =
                new ceres::QuaternionManifold;
            local_subset_homog_para_ptrs.push_back(quaternion_parameterization);
            problem_->SetManifold(qvec_data, quaternion_parameterization);
        }

        for (double *tvec_data : parameterized_tvec_data_)
        {
            ceres::Manifold *homogeneous_parameterization =
                new ceres::SphereManifold<3>();
            local_subset_homog_para_ptrs.push_back(homogeneous_parameterization);
            problem_->SetManifold(tvec_data, homogeneous_parameterization);
        }

        if (constant_camera)
        {
            for (auto &cam : corrImgs->cams)
            {
                problem_->SetParameterBlockConstant(cam.second.ParamsData());
            }
        }
        else
        {
            std::unordered_map<int, bool> constant_cams;
            // const size_t nr_cams = corrImgs->cams.size();
            if (corrImgs->constant_cams.empty())
            {
                for (const auto &cam : corrImgs->cams){
                    constant_cams.emplace(cam.first, false);
                }
            }
            else
            {
                constant_cams = corrImgs->constant_cams;
            }
            for (auto &cam : corrImgs->cams)
            {
                if (constant_cams.at(cam.first))
                {
                    problem_->SetParameterBlockConstant(cam.second.ParamsData());
                }
                else
                {
                    setFixedIntrinsics(cam.second);
                }
            }
        }
    }

    void RelativePoseBundleAdjuster::setFixedIntrinsics(colmap::Camera &cam)
    {
        std::vector<int> const_camera_params;

        if (!options_.refine_focal_length)
        {
            const std::vector<size_t> &params_idxs = cam.FocalLengthIdxs();
            const_camera_params.insert(const_camera_params.end(),
                                       params_idxs.begin(), params_idxs.end());
        }
        if (!options_.refine_principal_point)
        {
            const std::vector<size_t> &params_idxs = cam.PrincipalPointIdxs();
            const_camera_params.insert(const_camera_params.end(),
                                       params_idxs.begin(), params_idxs.end());
        }
        if (!options_.refine_extra_params)
        {
            const std::vector<size_t> &params_idxs = cam.ExtraParamsIdxs();
            if (!params_idxs.empty())
            {
                const_camera_params.insert(const_camera_params.end(),
                                           params_idxs.begin(), params_idxs.end());
            }
        }
        // else
        // {
        //     SetDistortionLowerUpperBounds(problem_, cam);
        // }

        if (const_camera_params.size() > 0)
        {
            ceres::Manifold *camera_params_parameterization =
                new ceres::SubsetManifold(
                    static_cast<int>(cam.NumParams()), const_camera_params);
            local_subset_homog_para_ptrs.push_back(camera_params_parameterization);
            problem_->SetManifold(cam.ParamsData(),
                                          camera_params_parameterization);
        }
    }

    FixedDepthBundleAdjuster::FixedDepthBundleAdjuster(BundleAdjustmentOptions options, const bool use_ceres_ownership, const bool allImgsNeeded, const bool allCamsNeeded)
        : allImgsNeeded_(allImgsNeeded), allCamsNeeded_(allCamsNeeded), options_(std::move(options)), use_ceres_ownership_(use_ceres_ownership)
    {
        CHECK(options_.Check());
        loss_function = options_.CreateLossFunction();
    }

    bool FixedDepthBundleAdjuster::Solve(FixedDepthBAData *corrImgs)
    {
        if (!corrImgs->Check())
        {
            return false;
        }
        ceres::Problem::Options problem_opts;
        if (use_ceres_ownership_)
        {
            problem_opts.enable_fast_removal = true;
            problem_opts.cost_function_ownership = ceres::TAKE_OWNERSHIP;
            problem_opts.disable_all_safety_checks = false;
            problem_opts.manifold_ownership = ceres::TAKE_OWNERSHIP;
        }
        else
        {
            problem_opts.enable_fast_removal = false;
            problem_opts.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_opts.disable_all_safety_checks = false;
            problem_opts.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        }
        problem_opts.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        if (problem_)
        {
            problem_.reset(new ceres::Problem(problem_opts));
        }
        else
        {
            problem_ = std::make_unique<ceres::Problem>(problem_opts);
        }
        parameterized_qvec_data_.clear();

        if (!SetUp(corrImgs))
        {
            std::cerr << "Setting up BA problem failed" << std::endl;
            return false;
        }

        if (problem_->NumResiduals() < 20)
        {
            std::cerr << "Too less residuals for BA remaining" << std::endl;
            return false;
        }

        ceres::Solver::Options solver_options = options_.solver_options;
        if (options_.useDefaultSolver){
            solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // ceres::DENSE_SCHUR;
            if (options_.refine_extra_params)
            {
                solver_options.trust_region_strategy_type = ceres::DOGLEG; //ceres::LEVENBERG_MARQUARDT
                solver_options.dogleg_type = ceres::SUBSPACE_DOGLEG;       //ceres::TRADITIONAL_DOGLEG
                solver_options.use_nonmonotonic_steps = true;
                solver_options.max_consecutive_nonmonotonic_steps = 6; //5
                solver_options.min_trust_region_radius = 1e-16;        //1e-32
            }
        }

        int num_threads = GetEffectiveNumThreads(solver_options.num_threads);
        if ((options_.CeresCPUcnt > 0) && (num_threads > options_.CeresCPUcnt))
        {
            num_threads = options_.CeresCPUcnt;
        }
        solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
        num_threads = GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
        if ((options_.CeresCPUcnt > 0) && (num_threads > options_.CeresCPUcnt))
        {
            num_threads = options_.CeresCPUcnt;
        }
        solver_options.num_linear_solver_threads = num_threads;
#endif // CERES_VERSION_MAJOR

        std::string solver_error;
        CHECK(solver_options.IsValid(&solver_error)); // << solver_error;

        ceres::Solve(solver_options, problem_.get(), &summary_);

        if (solver_options.minimizer_progress_to_stdout)
        {
            std::cout << std::endl;
        }

        if (options_.print_summary)
        {
            PrintHeading2("Fixed depth bundle adjustment report");
            PrintSolverSummary(summary_);
        }

        return summary_.IsSolutionUsable();
    }

    bool FixedDepthBundleAdjuster::SetUp(FixedDepthBAData *corrImgs)
    {
        AddImagesToProblem(corrImgs);
        if (refine_extrinsics && parameterized_qvec_data_.empty())
        {
            return false;
        }
        if (allCamsNeeded_ && used_cams.size() != corrImgs->cams.size())
        {
            return false;
        }
        ParameterizeCameras(corrImgs);
        return true;
    }

    void FixedDepthBundleAdjuster::AddImagesToProblem(FixedDepthBAData *corrImgs)
    {
        double max_squared_reproj_error = max_reproj_error * max_reproj_error;

        const size_t nr_imgs = corrImgs->imgs_to_cam_idx.size();
        q_used = std::vector<int>(corrImgs->points3d.size(), 1);
        for (size_t i = 0; i < nr_imgs; ++i)
        {
            const size_t cam_nr = corrImgs->imgs_to_cam_idx.at(i);
            colmap::Camera &cam = corrImgs->cams[cam_nr];

            const colmap::CamMatDampingSingle *camMatDampPars = corrImgs->camMat_damp ? &(corrImgs->camMatDampPars.at(cam_nr)) : nullptr;

            double *qvec_data = nullptr;
            double *tvec_data = nullptr;
            double *camera_params_data = cam.ParamsData();

            Eigen::Vector4d &quat = corrImgs->quats[i];
            Eigen::Vector3d &t = corrImgs->ts[i];

            // CostFunction assumes unit quaternions.
            quat = colmap::NormalizeQuaternion(quat);
            qvec_data = quat.data();
            tvec_data = t.data();

            bool pose_constant = !options_.refine_extrinsics;
            if (!corrImgs->constant_poses.empty())
            {
                pose_constant |= corrImgs->constant_poses.at(i);
            }

            // The number of added observations for the current image.
            size_t num_observations = 0;

            // Add residuals to bundle adjustment problem.
            const std::vector<Eigen::Vector2d> &corrs_this = corrImgs->corrs.at(i);
            const std::vector<size_t> &pt_3d_idx = corrImgs->points3d_idx.at(i);
            size_t idx = 0;
            for (const auto &corr : corrs_this)
            {
                const size_t &ptx_3d_idx = pt_3d_idx.at(idx++);
                const std::pair<int, double> img_depth = corrImgs->depth_vals_imgs.at(ptx_3d_idx);
                Eigen::Vector2d &ptx_depth_scale = corrImgs->depth_scales_imgs.at(img_depth.first);
                Eigen::Vector3d &pt_3d = corrImgs->points3d[ptx_3d_idx];
                if (corrImgs->calcReprojectionError(corr, pt_3d, img_depth.second, ptx_depth_scale, quat, t, cam) > max_squared_reproj_error)
                {
                    q_used[ptx_3d_idx]--;
                    continue;
                }
                q_used[ptx_3d_idx] = static_cast<int>(nr_imgs);

                num_observations += 1;

                double *point3d_data = pt_3d.data();
                double *scale_data = ptx_depth_scale.data();

                ceres::CostFunction *cost_function = nullptr;
                if (pose_constant)
                {
                    switch (cam.ModelId())
                    {
#define CAMERA_MODEL_CASE(CameraModel)                                                  \
    case CameraModel::kModelId:                                                         \
        cost_function =                                                                 \
            BAFixedDepthConstCamsCostFunction<CameraModel>::Create(quat,                \
                                                                   t,                   \
                                                                   img_depth.second,    \
                                                                   corr,                \
                                                                   corrImgs->dist_damp, \
                                                                   camMatDampPars);     \
                                                                                        \
        break;

                        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, point3d_data, scale_data, camera_params_data);
                }
                else
                {
                    switch (cam.ModelId())
                    {
#define CAMERA_MODEL_CASE(CameraModel)                                                                                  \
    case CameraModel::kModelId:                                                                                         \
        cost_function =                                                                                                 \
            BAFixedDepthCostFunction<CameraModel>::Create(img_depth.second, corr, corrImgs->dist_damp, camMatDampPars); \
                                                                                                                        \
        break;

                        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, scale_data, point3d_data, camera_params_data);
                }
                cost_func_ptrs.push_back(cost_function);
            }

            if (!pose_constant && num_observations > 0)
            {
                parameterized_qvec_data_.insert(qvec_data);
            }
            else if (num_observations == 0 && allImgsNeeded_)
            {
                parameterized_qvec_data_.clear();
                return;
            }
            if (num_observations > 0)
            {
                used_cams.emplace(cam_nr);
            }
            if (parameterized_qvec_data_.empty() && !used_cams.empty())
            {
                refine_extrinsics = false;
            }
        }
    }

    void FixedDepthBundleAdjuster::ParameterizeCameras(FixedDepthBAData *corrImgs)
    {
        const bool constant_camera = !options_.refine_focal_length &&
                                     !options_.refine_principal_point &&
                                     !options_.refine_extra_params;
        for (double *qvec_data : parameterized_qvec_data_)
        {
            ceres::Manifold *quaternion_parameterization =
                new ceres::QuaternionManifold;
            local_subset_para_ptrs.push_back(quaternion_parameterization);
            problem_->SetManifold(qvec_data, quaternion_parameterization);
        }

        if (constant_camera)
        {
            for (auto &cam : corrImgs->cams)
            {
                problem_->SetParameterBlockConstant(cam.ParamsData());
            }
        }
        else
        {
            std::vector<bool> constant_cams;
            const size_t nr_cams = corrImgs->cams.size();
            if (corrImgs->constant_cams.empty())
            {
                constant_cams = std::vector<bool>(nr_cams, false);
            }
            else
            {
                constant_cams = corrImgs->constant_cams;
            }
            for (size_t i = 0; i < nr_cams; ++i)
            {
                colmap::Camera &cam = corrImgs->cams[i];
                if (constant_cams.at(i))
                {
                    problem_->SetParameterBlockConstant(cam.ParamsData());
                }
                else
                {
                    setFixedIntrinsics(cam);
                }
            }
        }
    }

    void FixedDepthBundleAdjuster::setFixedIntrinsics(colmap::Camera &cam)
    {
        std::vector<int> const_camera_params;

        if (!options_.refine_focal_length)
        {
            const std::vector<size_t> &params_idxs = cam.FocalLengthIdxs();
            const_camera_params.insert(const_camera_params.end(),
                                       params_idxs.begin(), params_idxs.end());
        }
        if (!options_.refine_principal_point)
        {
            const std::vector<size_t> &params_idxs = cam.PrincipalPointIdxs();
            const_camera_params.insert(const_camera_params.end(),
                                       params_idxs.begin(), params_idxs.end());
        }
        if (!options_.refine_extra_params)
        {
            const std::vector<size_t> &params_idxs = cam.ExtraParamsIdxs();
            if (!params_idxs.empty())
            {
                const_camera_params.insert(const_camera_params.end(),
                                           params_idxs.begin(), params_idxs.end());
            }
        }
        // else
        // {
        //     SetDistortionLowerUpperBounds(problem_, cam);
        // }

        if (const_camera_params.size() > 0)
        {
            ceres::Manifold *camera_params_parameterization =
                new ceres::SubsetManifold(
                    static_cast<int>(cam.NumParams()), const_camera_params);
            local_subset_para_ptrs.push_back(camera_params_parameterization);
            problem_->SetManifold(cam.ParamsData(),
                                          camera_params_parameterization);
        }
    }

    void SetDistortionLowerUpperBounds(std::unique_ptr<ceres::Problem> &problem_, colmap::Camera &cam)
    {
        const std::vector<size_t> &params_idxs = cam.ExtraParamsIdxs();
        const size_t nrElems = params_idxs.size();
        const auto camID = cam.ModelId();
        if ((camID > 1 && camID < 5) || camID == 6 || (camID > 10 && camID < 14))
        {
            const double r2_lower = -10.;
            const double r2_upper = 10.;
            const double r4_lower = -5.;
            const double r4_upper = 5.;
            const double r6_lower = -2.;
            const double r6_upper = 2.;
            switch (nrElems)
            {
            case 1:
                problem_->SetParameterLowerBound(cam.ParamsData(), 0, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 0, r2_upper);
                break;
            case 2:
                problem_->SetParameterLowerBound(cam.ParamsData(), 0, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 0, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 1, r4_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 1, r4_upper);
                break;
            case 4:
                problem_->SetParameterLowerBound(cam.ParamsData(), 0, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 0, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 1, r4_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 1, r4_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 2, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 2, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 3, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 3, r2_upper);
                break;
            case 5:
                problem_->SetParameterLowerBound(cam.ParamsData(), 0, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 0, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 1, r4_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 1, r4_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 2, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 2, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 3, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 3, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 4, r6_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 4, r6_upper);
                break;
            case 8:
                problem_->SetParameterLowerBound(cam.ParamsData(), 0, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 0, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 1, r4_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 1, r4_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 2, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 2, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 3, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 3, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 4, r6_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 4, r6_upper);

                problem_->SetParameterLowerBound(cam.ParamsData(), 5, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 5, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 6, r4_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 6, r4_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 7, r6_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 7, r6_upper);
                break;
            case 12:
            case 14:
                problem_->SetParameterLowerBound(cam.ParamsData(), 0, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 0, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 1, r4_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 1, r4_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 2, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 2, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 3, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 3, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 4, r6_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 4, r6_upper);

                problem_->SetParameterLowerBound(cam.ParamsData(), 5, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 5, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 6, r4_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 6, r4_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 7, r6_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 7, r6_upper);

                problem_->SetParameterLowerBound(cam.ParamsData(), 8, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 8, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 9, r4_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 9, r4_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 10, r2_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 10, r2_upper);
                problem_->SetParameterLowerBound(cam.ParamsData(), 11, r4_lower);
                problem_->SetParameterUpperBound(cam.ParamsData(), 11, r4_upper);
                break;
            default:
                break;
            }
        }
    }

    void PrintSolverSummary(const ceres::Solver::Summary &summary)
    {
        std::cout << std::right << std::setw(16) << "Residuals : ";
        std::cout << std::left << summary.num_residuals_reduced << std::endl;

        std::cout << std::right << std::setw(16) << "Parameters : ";
        std::cout << std::left << summary.num_effective_parameters_reduced
                  << std::endl;

        std::cout << std::right << std::setw(16) << "Iterations : ";
        std::cout << std::left
                  << summary.num_successful_steps + summary.num_unsuccessful_steps
                  << std::endl;

        std::cout << std::right << std::setw(16) << "Time : ";
        std::cout << std::left << summary.total_time_in_seconds << " [s]"
                  << std::endl;

        std::cout << std::right << std::setw(16) << "Initial cost : ";
        std::cout << std::right << std::setprecision(6)
                  << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
                  << " [px]" << std::endl;

        std::cout << std::right << std::setw(16) << "Final cost : ";
        std::cout << std::right << std::setprecision(6)
                  << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
                  << " [px]" << std::endl;

        std::cout << std::right << std::setw(16) << "Termination : ";

        std::string termination = "";

        switch (summary.termination_type)
        {
        case ceres::CONVERGENCE:
            termination = "Convergence";
            break;
        case ceres::NO_CONVERGENCE:
            termination = "No convergence";
            break;
        case ceres::FAILURE:
            termination = "Failure";
            break;
        case ceres::USER_SUCCESS:
            termination = "User success";
            break;
        case ceres::USER_FAILURE:
            termination = "User failure";
            break;
        default:
            termination = "Unknown";
            break;
        }

        std::cout << std::right << termination << std::endl;
        std::cout << std::endl;
    }

    void PrintHeading1(const std::string &heading)
    {
        std::cout << std::endl
                  << std::string(78, '=') << std::endl;
        std::cout << heading << std::endl;
        std::cout << std::string(78, '=') << std::endl
                  << std::endl;
    }

    void PrintHeading2(const std::string &heading)
    {
        std::cout << std::endl
                  << heading << std::endl;
        std::cout << std::string(std::min<int>(heading.size(), 78), '-') << std::endl;
    }

    int GetEffectiveNumThreads(const int num_threads)
    {
        int num_effective_threads = num_threads;
        if (num_threads <= 0)
        {
            num_effective_threads = std::thread::hardware_concurrency();
        }

        if (num_effective_threads <= 0)
        {
            num_effective_threads = 1;
        }

        return num_effective_threads;
    }
}