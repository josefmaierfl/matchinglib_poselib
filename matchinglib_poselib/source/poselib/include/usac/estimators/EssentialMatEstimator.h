//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2019 AIT Austrian Institute of Technology GmbH
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

#ifndef ESSENTIALMATESTIMATOR_H
#define ESSENTIALMATESTIMATOR_H


#include <iostream>
#include <fstream>
#include <string>
#include "usac/config/ConfigParamsEssentialMat.h"
#include "usac/utils/MathFunctions.h"
#include "usac/utils/FundmatrixFunctions.h"
#include "usac/utils/HomographyFunctions.h"
#include "usac/utils/PoseFunctions.h"
#include "usac/estimators/USAC.h"
#include "usac/utils/weightingEssential.h"

#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeWeightingAdapter.hpp>
#include <opengv/triangulation/methods.hpp>
#include <memory>

#include "poselib/pose_estim.h"
#include "poselib/pose_helper.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/eigen.hpp"

class EssentialMatEstimator : public USAC<EssentialMatEstimator>
{
public:
    inline bool		 initProblem(const ConfigParamsEssential& cfg, double* pointData);
    // ------------------------------------------------------------------------
    // storage for the final result
    std::vector<double> final_model_params_;
    std::vector<double> degen_final_model_params_;
    std::vector<double> degen_final_model_params_rot;
    std::vector<double> degen_final_model_params_trans;
    opengv::rotation_t R_eigen;							// Stores a rotation estimated using Kneip's Eigen solver
    opengv::translation_t t_eigen;						// Stores a translation estimated using Kneip's Eigen solver
    opengv::rotation_t R_eigen_degen;					// Stores a rotation (dedected during degeneracy test)
    opengv::translation_t t_eigen_upgrade;				// Stores an upgraded translation based on inliers and outliers from degenerate model 'no Motion'

public:
    EssentialMatEstimator()
    {
        input_points_ = nullptr;
        data_matrix_ = nullptr;
        degen_data_matrix_ = nullptr;
        models_.clear();
        models_denorm_.clear();
    };
    ~EssentialMatEstimator()
    {
        if (input_points_) { delete[] input_points_; input_points_ = nullptr; }
        if (data_matrix_) { delete[] data_matrix_; data_matrix_ = nullptr; }
        if (degen_data_matrix_) { delete[] degen_data_matrix_; degen_data_matrix_ = nullptr; }
        for (auto &i : models_)
        {
            delete[] i;
        }
        models_.clear();
        for (auto &i : models_denorm_)
        {
            delete[] i;
        }
        models_denorm_.clear();
        adapter.reset();
        adapter_denorm.reset();
    };

public:
    // ------------------------------------------------------------------------
    // problem specific functions
    inline void		 cleanupProblem();
    inline unsigned int generateMinimalSampleModels() override;
    inline bool		 generateRefinedModel(std::vector<unsigned int>& sample, const unsigned int numPoints,
        bool weighted = false, double* weights = nullptr) override;
    inline bool		 validateSample() override;
    inline bool		 validateModel(unsigned int modelIndex) override;
    inline bool		 evaluateModel(unsigned int modelIndex, unsigned int* numInliers, unsigned int* numPointsTested) override;
    inline void		 testSolutionDegeneracy(bool* degenerateModel, bool* upgradeModel) override;
    inline unsigned int upgradeDegenerateModel() override;
    inline void		 findWeights(unsigned int modelIndex, const std::vector<unsigned int>& inliers,
        unsigned int numInliers, double* weights) override;
    inline void		 storeModel(unsigned int modelIndex, unsigned int numInliers) override;

private:
    inline void testSolutionDegeneracyH(bool* degenerateModel, bool* upgradeModel);
    inline void testSolutionDegeneracyRot(bool* degenerateModel);
    inline void testSolutionDegeneracyTrans(bool* degenerateModel);
    inline void testSolutionDegeneracyNoMot(bool* degenerateModel);
    inline bool evaluateModelRot(const opengv::rotation_t &model, unsigned int* numInliers, unsigned int* numPointsTested);
    inline bool evaluateModelTrans(const opengv::translation_t &model, unsigned int* numInliers, unsigned int* numPointsTested);

private:
    double*		 input_points_denorm_;					    // stores pointer to original input points
    double       degen_homog_threshold_;				    // threshold for h-degen test
    double		 poseDegenTheshold;							// angular threshold for testing the degeneracy rotation only
    unsigned int degen_max_upgrade_samples_;				// maximum number of upgrade attempts for H
    unsigned int degen_max_upgrade_samples_rot;				// maximum number of upgrade attempts for R
    //unsigned int degen_max_upgrade_samples_trans;			// maximum number of upgrade attempts for t
    //unsigned int degen_max_upgrade_samples_noMot_rot;		// maximum number of upgrade attempts for no motion to R
    unsigned int degen_max_upgrade_samples_noMot_trans;		// maximum number of upgrade attempts for no motion to t
    opengv::rotation_t R_eigen_new;							// Stores temporally R_eigen
    opengv::translation_t t_eigen_new;						// Stores temporally t_eigen

    // ------------------------------------------------------------------------
    // temporary storage
    double* input_points_;								// stores normalized data points
    double* data_matrix_;								// linearized input data
    double* degen_data_matrix_;							// only for degeneracy testing
    std::vector<int> degen_outlier_flags_;				// for easy access to outliers to degeneracy
    std::vector<int> degen_outlier_flags_rot;			// outliers to rotatiobal degeneracy
    std::vector<int> degen_outlier_flags_trans;			// outliers to translational degeneracy
    std::vector<int> degen_outlier_flags_noMot;			// outliers to no motion degeneracy
    double  m_T1_[9], m_T2_[9], m_T2_trans_[9];			// normalization matrices
    Eigen::Matrix3d m_T1_e, m_T2_e, m_T2_trans_e;		// normalization matrices in Eigen format
    Eigen::Matrix3d m_T1_e_inv, m_T2_e_inv, m_T2_trans_e_inv;		// inverse normalization matrices in Eigen format
    opengv::essentials_t fivept_nister_essentials;		// stores vector of models
    opengv::essentials_t fivept_nister_essentials_denorm; // stores vector of (denormalized) models
    std::vector<double*> models_;						// stores vector of models
    std::vector<double*> models_denorm_;				// stores vector of (denormalized) models

    std::unique_ptr<opengv::relative_pose::CentralRelativeAdapter> adapter; //Pointer to adapter for OpenGV
    std::shared_ptr<opengv::relative_pose::CentralRelativeAdapter> adapter_denorm; //Pointer to adapter for OpenGV with denormalized correspondences
    //std::shared_ptr<opengv::relative_pose::CentralRelativeAdapter> adapter_denorm_kneip; //Pointer to adapter for OpenGV if Kneip's Eigen solver is used
    opengv::bearingVectors_t bearingVectors1_all;
    opengv::bearingVectors_t bearingVectors2_all;
    opengv::bearingVectors_t bearingVectors1_denorm;
    opengv::bearingVectors_t bearingVectors2_denorm;
    USACConfig::RefineAlgorithm refineMethod; //Used method within generateRefinedModel
    USACConfig::EssentialMatEstimatorUsed used_estimator;//Specifies, which estimator from OpenGV should be used (Nister, Stewenius, Kneip's Eigen)
    cv::Mat p1, p2;
    enum PoseDegeneracy
    {
        DEGEN_NOT_FOUND = 0x1,
        DEGEN_H = 0x2,
        DEGEN_ROT_TRANS = 0x4,
        DEGEN_NO_MOT = 0x8,
        DEGEN_UPGRADE = 0x10
    };
    unsigned int degeneracyType;
    bool ransacLikeUpgradeDegenerate;
    bool enableHDegen;
    bool enableUpgradeDegenPose;
    std::vector<int> degen_sample_rot;
    std::vector<int> degen_sample_trans;
    std::vector<int> degen_sample_noMot;
};

// ============================================================================================
// initProblem: initializes problem specific data and parameters
// this function is called once per run on new data
// ============================================================================================
bool EssentialMatEstimator::initProblem(const ConfigParamsEssential& cfg, double* pointData)
{
    // copy pointer to input data
    input_points_denorm_ = pointData;
    input_points_ = new double[6 * cfg.common.numDataPoints];
    if (input_points_denorm_ == nullptr)
    {
        std::cerr << "Input point data not properly initialized" << std::endl;
        return false;
    }
    if (input_points_ == nullptr)
    {
        std::cerr << "Could not allocate storage for normalized data points" << std::endl;
        return false;
    }

    // normalize input data
    // following this, input_points_ has the normalized points and input_points_denorm_ has
    // the original input points
    FTools::normalizePoints(input_points_denorm_, input_points_, cfg.common.numDataPoints, m_T1_, m_T2_);
    MathTools::mattr(m_T2_trans_, m_T2_, 3, 3);
    //store normalization matrices into Eigen matrices
    for (unsigned int i = 0; i < 3; i++)
    {
        for (unsigned int j = 0; j < 3; j++)
        {
            m_T1_e(i, j) = m_T1_[3 * i + j];
            m_T2_e(i, j) = m_T2_[3 * i + j];
            m_T2_trans_e(i, j) = m_T2_trans_[3 * i + j];
        }
    }
    m_T1_e_inv = m_T1_e.inverse();
    m_T2_e_inv = m_T2_e.inverse();
    m_T2_trans_e_inv = m_T2_trans_e.inverse();

    // allocate storage for models
    final_model_params_.clear();
    final_model_params_.resize(9);
    for (auto &i : models_)
    {
        delete[] i;
    }
    models_.clear();
    models_.resize(usac_max_solns_per_sample_);
    for (auto &i : models_denorm_)
    {
        delete[] i;
    }
    models_denorm_.clear();
    models_denorm_.resize(usac_max_solns_per_sample_);
    for (unsigned int i = 0; i < usac_max_solns_per_sample_; ++i)
    {
        models_[i] = new double[9];
        models_denorm_[i] = new double[9];
    }
    R_eigen = opengv::rotation_t::Identity();
    t_eigen = opengv::translation_t::Zero();
    R_eigen_new = opengv::rotation_t::Identity();
    t_eigen_new = opengv::translation_t::Zero();

    p1 = cv::Mat(cfg.common.numDataPoints, 2, CV_64FC1);
    p2 = cv::Mat(cfg.common.numDataPoints, 2, CV_64FC1);
    double* p_idx1 = input_points_denorm_;
    double* p_idx2 = input_points_;
    for (unsigned int i = 0; i < cfg.common.numDataPoints; i++)
    {
        opengv::point_t bodyPoint1;// (input_points_ + i * 6);
        opengv::point_t bodyPoint2;// (input_points_ + i * 6 + 3);
        bodyPoint1 << *(p_idx2), *(p_idx2 + 1), *(p_idx2 + 2);
        bodyPoint2 << *(p_idx2 + 3), *(p_idx2 + 4), *(p_idx2 + 5);

        p1.at<double>(i, 0) = *(p_idx1);
        p1.at<double>(i, 1) = *(p_idx1 + 1);
        p2.at<double>(i, 0) = *(p_idx1 + 3);
        p2.at<double>(i, 1) = *(p_idx1 + 4);
        p_idx2 = p_idx2 + 6;

        bodyPoint1 = bodyPoint1 / bodyPoint1.norm();
        bodyPoint2 = bodyPoint2 / bodyPoint2.norm();
        bearingVectors1_all.push_back(bodyPoint1);
        bearingVectors2_all.push_back(bodyPoint2);

        opengv::point_t bodyPoint1_d;// (input_points_denorm_ + i * 6);
        opengv::point_t bodyPoint2_d;// (input_points_denorm_ + i * 6 + 3);
        bodyPoint1_d << *(p_idx1), *(p_idx1 + 1), *(p_idx1 + 2);
        bodyPoint2_d << *(p_idx1 + 3), *(p_idx1 + 4), *(p_idx1 + 5);
        p_idx1 = p_idx1 + 6;
        bodyPoint1_d = bodyPoint1_d / bodyPoint1_d.norm();
        bodyPoint2_d = bodyPoint2_d / bodyPoint2_d.norm();
        bearingVectors1_denorm.push_back(bodyPoint1_d);
        bearingVectors2_denorm.push_back(bodyPoint2_d);


    }

    //create a central relative adapter
    //opengv inverts the input inside the function (as Nister and Stewenius deliver a inverted pose), so invert the input here as well!!!!!!!!!!!
    adapter.reset(new opengv::relative_pose::CentralRelativeAdapter(
        bearingVectors2_all,
        bearingVectors1_all));

    adapter_denorm.reset(new opengv::relative_pose::CentralRelativeAdapter(
        bearingVectors2_denorm,
        bearingVectors1_denorm));

    /*adapter_denorm_kneip.reset(new opengv::relative_pose::CentralRelativeAdapter(
        bearingVectors1_denorm,
        bearingVectors2_denorm));*/

    fivept_nister_essentials = opengv::essentials_t(usac_max_solns_per_sample_);
    fivept_nister_essentials_denorm = opengv::essentials_t(usac_max_solns_per_sample_);


    // precompute the data matrix
    data_matrix_ = new double[9 * usac_num_data_points_];	// 9 values per correspondence
    FTools::computeDataMatrix(data_matrix_, usac_num_data_points_, input_points_);

    // if required, set up H-data matrix/storage for degeneracy testing
    degen_outlier_flags_.clear(); degen_outlier_flags_.resize(usac_num_data_points_, 0);
    degeneracyType = DEGEN_NOT_FOUND;
    ransacLikeUpgradeDegenerate = cfg.fund.ransacLikeUpgradeDegenerate;
    enableUpgradeDegenPose = cfg.fund.enableUpgradeDegenPose;
    enableHDegen = cfg.fund.enableHDegen;
    if (usac_test_degeneracy_)//---------------------------------------------------> update below
    {
        degen_homog_threshold_ = cfg.fund.hDegenThreshold;
        degen_max_upgrade_samples_ = cfg.fund.maxUpgradeSamples;
        degen_max_upgrade_samples_rot = cfg.fund.maxUpgradeSamples;
        //degen_max_upgrade_samples_trans = cfg.fund.maxUpgradeSamples;
        //degen_max_upgrade_samples_noMot_rot = cfg.fund.maxUpgradeSamples;
        degen_max_upgrade_samples_noMot_trans = cfg.fund.maxUpgradeSamples;
        degen_final_model_params_.clear(); degen_final_model_params_.resize(9);
        degen_final_model_params_rot.clear(); degen_final_model_params_rot.resize(9);
        degen_final_model_params_trans.clear(); degen_final_model_params_trans.resize(3);
        degen_data_matrix_ = new double[2 * 9 * usac_num_data_points_];	// 2 equations per correspondence
        HTools::computeDataMatrix(degen_data_matrix_, usac_num_data_points_, input_points_);
        degen_outlier_flags_rot.resize(usac_num_data_points_);
        //degen_outlier_flags_trans.resize(usac_num_data_points_);;
        degen_outlier_flags_noMot.resize(usac_num_data_points_);
        degen_sample_trans.resize(2);
    }
    else
    {
        degen_homog_threshold_ = 0.0;
        degen_max_upgrade_samples_ = 0;
        degen_max_upgrade_samples_rot = 0;
        //degen_max_upgrade_samples_trans = 0;
        //degen_max_upgrade_samples_noMot_rot = 0;
        degen_max_upgrade_samples_noMot_trans = 0;
        degen_final_model_params_.clear();
        degen_final_model_params_rot.clear();
        degen_final_model_params_trans.clear();
        degen_data_matrix_ = nullptr;
    }

    // read in the f-matrix specific parameters from the config struct
    //matrix_decomposition_method = cfg.fund.decompositionAlg;
    refineMethod = cfg.fund.refineMethod;
    used_estimator = cfg.fund.used_estimator;
    degen_homog_threshold_ = cfg.fund.hDegenThreshold*cfg.fund.hDegenThreshold;
    poseDegenTheshold = 1.0 - std::cos(std::atan(cfg.fund.rotDegenThesholdPix / cfg.fund.focalLength));

    return true;
}


// ============================================================================================
// cleanupProblem: release any temporary problem specific data storage
// this function is called at the end of each run on new data
// ============================================================================================
void EssentialMatEstimator::cleanupProblem()
{
    if (input_points_) { delete[] input_points_; input_points_ = nullptr; }
    if (data_matrix_) { delete[] data_matrix_; data_matrix_ = nullptr; }
    if (degen_data_matrix_) { delete[] degen_data_matrix_; degen_data_matrix_ = nullptr; }
    for (auto &i : models_)
    {
        delete[] i;
    }
    models_.clear();
    for (auto &i : models_denorm_)
    {
        delete[] i;
    }
    models_denorm_.clear();
    degen_outlier_flags_.clear();
}


// ============================================================================================
// generateMinimalSampleModels: generates minimum sample model(s) from the data points whose
// indices are currently stored in m_sample.
// the generated models are stored in a vector of models and are all evaluated
// ============================================================================================
unsigned int EssentialMatEstimator::generateMinimalSampleModels()
{
    unsigned int nsols = 0;
    std::vector<int> indices;
    for (unsigned int i = 0; i < usac_min_sample_size_; ++i)
    {
        indices.push_back((int)min_sample_[i]);
    }
    if (used_estimator == USACConfig::ESTIM_NISTER)
    {
        fivept_nister_essentials_denorm.clear();
        fivept_nister_essentials_denorm = opengv::relative_pose::fivept_nister(*adapter_denorm, indices);
        nsols = fivept_nister_essentials_denorm.size();
        if (nsols > usac_max_solns_per_sample_){
            return 0;
        }
//        CV_Assert(nsols <= usac_max_solns_per_sample_);
    }
    else if (used_estimator == USACConfig::ESTIM_EIG_KNEIP)
    {
        opengv::eigensolverOutput_t eig_out;
        opengv::essential_t E_eigen;
        cv::Mat R_tmp, t_tmp, E_tmp;
        fivept_nister_essentials_denorm.clear();
        nsols = MAX_SOLS_KNEIP;
        for (unsigned int i = 0; i < MAX_SOLS_KNEIP; i++)
        {
            //Variation of R as init for eigen-solver
            opengv::rotation_t R_init = Eigen::Matrix3d::Identity();
            PoseTools::getPerturbedRotation(R_init, RAND_ROTATION_AMPLITUDE); //Check if the amplitude is too large or too small!

            adapter_denorm->setR12(R_init);
            eig_out.rotation = R_init;
            R_eigen = opengv::relative_pose::eigensolver(*adapter_denorm, indices, eig_out);
            t_eigen = eig_out.translation;
            t_eigen /= t_eigen.norm();
            bool kneipFailed = false;
            for (size_t r = 0; r < 3; r++)
            {
                for (size_t c = 0; c < 3; c++)
                {
                    if (isnan(((long double) R_eigen(r, c))))
                    {
                        nsols--;
                        kneipFailed = true;
                        break;
                    }
                }
                if (kneipFailed)
                    break;
            }
            if (kneipFailed)
                continue;
            /*if ((poselib::nearZero(R_eigen(0, 0))
            || poselib::nearZero(R_eigen(1, 1))
            || poselib::nearZero(R_eigen(2, 2)))) {
                nsols--;
                continue;
            }*/
            if(!poselib::isMatRoationMat(R_eigen)){
                nsols--;
                continue;
            }
            cv::eigen2cv(R_eigen, R_tmp);
            cv::eigen2cv(t_eigen, t_tmp);
            E_tmp = poselib::getEfromRT(R_tmp, t_tmp);
            cv::cv2eigen(E_tmp, E_eigen);
            fivept_nister_essentials_denorm.push_back(E_eigen);
        }
        if (nsols > usac_max_solns_per_sample_){
            return 0;
        }
//        CV_Assert(nsols <= usac_max_solns_per_sample_);
        if (nsols == 0)
            return 0;
    }
    else if (used_estimator == USACConfig::ESTIM_STEWENIUS)
    {
        opengv::complexEssentials_t fivept_stewenius_essentials;
        fivept_stewenius_essentials = opengv::relative_pose::fivept_stewenius(*adapter_denorm, indices);
        fivept_nister_essentials_denorm.clear();
        //Skip essential matrices with imaginary entries
        for (auto &i : fivept_stewenius_essentials)
        {
            bool is_Imag = false;
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    if (!poselib::nearZero(100 * i(r, c).imag()))
                    {
                        is_Imag = true;
                        break;
                    }
                }
                if (is_Imag)
                    break;
            }
            if (!is_Imag)
            {
                opengv::essential_t E_eigen;
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        E_eigen(r, c) = i(r, c).real();
                fivept_nister_essentials_denorm.push_back(E_eigen);
            }
        }

        nsols = fivept_nister_essentials_denorm.size();
        if (nsols > usac_max_solns_per_sample_){
            return 0;
        }
//        CV_Assert(nsols <= usac_max_solns_per_sample_);
    }
    else
    {
        std::cout << "Estimator not supported! Using Nister." << std::endl;
        fivept_nister_essentials_denorm = opengv::relative_pose::fivept_nister(*adapter_denorm, indices);
        nsols = fivept_nister_essentials_denorm.size();
        if (nsols > usac_max_solns_per_sample_){
            return 0;
        }
//        CV_Assert(nsols <= usac_max_solns_per_sample_);
    }

    fivept_nister_essentials.clear();
    for (unsigned int i = 0; i < nsols; ++i)
    {
        fivept_nister_essentials.push_back(m_T2_trans_e_inv * fivept_nister_essentials_denorm.at(i) * m_T1_e_inv);
        //Store result back into USAC format to allow using their validateSample() code
        for (size_t r = 0; r < 3; r++)
            for (size_t c = 0; c < 3; c++)
            {
                *(models_[i] + r * 3 + c) = fivept_nister_essentials[i](r, c);
                *(models_denorm_[i] + r * 3 + c) = fivept_nister_essentials_denorm[i](r, c);
            }
    }

    return nsols;
}


// ============================================================================================
// generateRefinedModel: compute model using non-minimal set of samples
// default operation is to use a weight of 1 for every data point
// ============================================================================================
bool EssentialMatEstimator::generateRefinedModel(std::vector<unsigned int>& sample,
    const unsigned int numPoints,
    bool weighted,
    double* weights)
{
    if (numPoints < usac_min_sample_size_){
        return false;
    }
    /*if (posetype == USACConfig::TRANS_ESSENTIAL)
    {*/
        if (refineMethod == USACConfig::REFINE_WEIGHTS)
        {
            // form the matrix of equations for this non-minimal sample
            auto *A = new double[numPoints * 9];
            double *src_ptr;
            double *dst_ptr = A;
            for (unsigned int i = 0; i < numPoints; ++i)
            {
                src_ptr = data_matrix_ + sample[i];
                for (unsigned int j = 0; j < 9; ++j)
                {
                    if (!weighted)
                    {
                        *dst_ptr = *src_ptr;
                    }
                    else
                    {
                        *dst_ptr = (*src_ptr)*weights[i];
                    }
                    ++dst_ptr;
                    src_ptr += usac_num_data_points_;
                }
            }

            double Cv[9 * 9];
            FTools::formCovMat(Cv, A, numPoints, 9);

            double V[9 * 9], D[9], *p;
            MathTools::svdu1v(D, Cv, 9, V, 9);

            unsigned int j = 0;
            for (unsigned int i = 1; i < 9; ++i)
            {
                if (D[i] < D[j])
                {
                    j = i;
                }
            }
            p = V + j;

            for (unsigned int i = 0; i < 9; ++i)
            {
                *(models_[0] + i) = *p;
                p += 9;
            }
            FTools::singulF(models_[0]);
            // store denormalized version as well
            double T2_F[9];
            MathTools::mmul(T2_F, m_T2_trans_, models_[0], 3);
            MathTools::mmul(models_denorm_[0], T2_F, m_T1_, 3);

            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                {
                    fivept_nister_essentials[0](r, c) = *(models_[0] + r * 3 + c);
                    fivept_nister_essentials_denorm[0](r, c) = *(models_denorm_[0] + r * 3 + c);
                }

            delete[] A;
        }
        else if (refineMethod == USACConfig::REFINE_8PT_PSEUDOHUBER)
        {
            std::vector<int> indices;
            for (unsigned int i = 0; i < numPoints; ++i)
            {
                indices.push_back((int)sample[i]);
            }

            if (weighted)
            {
                cv::Mat E(3, 3, CV_64FC1);
                cv::eigen2cv(fivept_nister_essentials_denorm[0], E);
                cv::Mat mask = cv::Mat::zeros(1, (int)usac_num_data_points_, CV_8U);
                double pseudohuberth = /*m_T1_[0] **/ sqrt(usac_inlier_threshold_) / 10.0;
                for (unsigned int i = 0; i < numPoints; ++i)
                {
                    mask.at<bool>((int)sample[i]) = true;
                }
                poselib::robustEssentialRefine(p1, p2, E, E, pseudohuberth, 1, true, nullptr, nullptr, cv::noArray(), mask, 0, false, false);
                cv::cv2eigen(E, fivept_nister_essentials_denorm[0]);
                //fivept_nister_essentials_denorm[0] = m_T2_trans_e * fivept_nister_essentials[0] * m_T1_e;
                fivept_nister_essentials[0] = m_T2_trans_e_inv * fivept_nister_essentials_denorm[0] * m_T1_e_inv;
            }
            else
            {
                fivept_nister_essentials[0] = opengv::relative_pose::eightpt(*adapter, indices);
                //poselib::getClosestE(fivept_nister_essentials[0]);
                fivept_nister_essentials_denorm[0] = m_T2_trans_e * fivept_nister_essentials[0] * m_T1_e;
            }
            //Store result back into USAC format to allow using their validateSample() code
            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                {
                    *(models_[0] + r * 3 + c) = fivept_nister_essentials[0](r, c);
                    *(models_denorm_[0] + r * 3 + c) = fivept_nister_essentials_denorm[0](r, c);
                }
        }
        else if (refineMethod == USACConfig::REFINE_STEWENIUS || refineMethod == USACConfig::REFINE_STEWENIUS_WEIGHTS)
        {
            opengv::complexEssentials_t comlexEs;
            opengv::essentials_t Es_eigen;
            opengv::essential_t E_singele;

            if (refineMethod == USACConfig::REFINE_STEWENIUS_WEIGHTS && weighted)
            {
                opengv::bearingVectors_t bearingVectors1;
                opengv::bearingVectors_t bearingVectors2;
                std::vector<double> weights_vec;
                std::unique_ptr<opengv::relative_pose::CentralRelativeWeightingAdapter> adapter_denorm_weights; //Pointer to adapter for OpenGV if Stewenius with weights is used
                for (unsigned int i = 0; i < numPoints; i++)
                {
                    bearingVectors1.push_back(adapter_denorm->getBearingVector1(sample[i]));
                    bearingVectors2.push_back(adapter_denorm->getBearingVector2(sample[i]));
                    weights_vec.push_back((weights[i]));
                }
                adapter_denorm_weights.reset(new opengv::relative_pose::CentralRelativeWeightingAdapter(
                    bearingVectors1,
                    bearingVectors2,
                    weights_vec));

                comlexEs = fivept_stewenius_weight(*adapter_denorm_weights);
            }
            else
            {
                std::vector<int> indices;
                for (unsigned int i = 0; i < numPoints; ++i)
                {
                    indices.push_back((int)sample[i]);
                }
                comlexEs = opengv::relative_pose::fivept_stewenius(*adapter_denorm, indices);
            }

            for (auto &i : comlexEs)
            {
                bool is_Imag = false;
                for (int r = 0; r < 3; r++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        if (!poselib::nearZero(100 * i(r, c).imag()))
                        {
                            is_Imag = true;
                            break;
                        }
                    }
                    if (is_Imag)
                        break;
                }
                if (!is_Imag)
                {
                    opengv::essential_t E_eigen;
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            E_eigen(r, c) = i(r, c).real();
                    Es_eigen.push_back(E_eigen);
                }
            }

            size_t nsols = Es_eigen.size();
            if (nsols > 1)
            {
                /*std::vector<double> E_diffs(nsols);
                opengv::essential_t E_minSample;
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        E_minSample(r,c) = final_model_params_[r * 3 + c];
                E_minSample /= E_minSample.norm();
                for (size_t i = 0; i < nsols; i++)
                {
                    opengv::essential_t E_tmp = Es_eigen[i] / Es_eigen[i].norm();
                    E_tmp -= E_minSample;
                    E_diffs[i] = E_tmp.norm();
                }
                int min_elem = std::distance(E_diffs.begin(), std::min_element(E_diffs.begin(), E_diffs.end()));*/

                std::vector<double> errSums(nsols, 0);
                std::vector<std::vector<double>> possible_models(nsols);
                for (size_t j = 0; j < nsols; j++)
                {
                    possible_models[j].resize(9);
                    for (size_t r = 0; r < 3; r++)
                        for (size_t c = 0; c < 3; c++)
                            possible_models[j][r * 3 + c] = Es_eigen[j](r, c);
                }
                for (unsigned int i = 0; i < usac_num_data_points_; ++i)
                {
                    if (usac_results_.inlier_flags_[i])
                    {
                        for (size_t j = 0; j < nsols; j++)
                        {
                            errSums[j] += PoseTools::getSampsonError(possible_models[j], input_points_denorm_, i);
                        }
                        if ((i > 3) && (i % 4 == 0))
                        {
                            std::vector<double> errSums_tmp = errSums;
                            std::partial_sort(errSums_tmp.begin(), errSums_tmp.begin() + 2, errSums_tmp.end());
                            if (errSums_tmp[0] < 0.66 * errSums_tmp[1])
                            {
                                break;
                            }
                        }
                    }
                }
                int min_elem = std::distance(errSums.begin(), std::min_element(errSums.begin(), errSums.end()));
                E_singele = Es_eigen[min_elem];
            }
            else if (nsols == 1)
                E_singele = Es_eigen[0];
            else
                return false;

            fivept_nister_essentials_denorm[0] = E_singele;
            fivept_nister_essentials[0] = m_T2_trans_e_inv * E_singele * m_T1_e_inv;
            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                {
                    *(models_[0] + r * 3 + c) = fivept_nister_essentials[0](r, c);
                    *(models_denorm_[0] + r * 3 + c) = fivept_nister_essentials_denorm[0](r, c);
                }
        }
        else if (refineMethod == USACConfig::REFINE_NISTER || refineMethod == USACConfig::REFINE_NISTER_WEIGHTS)
        {
            opengv::essentials_t Es_eigen;
            opengv::essential_t E_singele;

            if (refineMethod == USACConfig::REFINE_NISTER_WEIGHTS && weighted)
            {
                opengv::bearingVectors_t bearingVectors1;
                opengv::bearingVectors_t bearingVectors2;
                std::vector<double> weights_vec;
                std::unique_ptr<opengv::relative_pose::CentralRelativeWeightingAdapter> adapter_denorm_weights; //Pointer to adapter for OpenGV if Stewenius with weights is used
                for (unsigned int i = 0; i < numPoints; i++)
                {
                    bearingVectors1.push_back(adapter_denorm->getBearingVector1(sample[i]));
                    bearingVectors2.push_back(adapter_denorm->getBearingVector2(sample[i]));
                    weights_vec.push_back((weights[i]));
                }
                adapter_denorm_weights.reset(new opengv::relative_pose::CentralRelativeWeightingAdapter(
                    bearingVectors1,
                    bearingVectors2,
                    weights_vec));

                Es_eigen = fivept_nister_weight(*adapter_denorm_weights);
            }
            else
            {
                std::vector<int> indices;
                for (unsigned int i = 0; i < numPoints; ++i)
                {
                    indices.push_back((int)sample[i]);
                }
                Es_eigen = opengv::relative_pose::fivept_nister(*adapter_denorm, indices);
            }

            size_t nsols = Es_eigen.size();
            if (nsols > 1)
            {
                /*std::vector<double> E_diffs(nsols);
                opengv::essential_t E_minSample;
                for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                E_minSample(r,c) = final_model_params_[r * 3 + c];
                E_minSample /= E_minSample.norm();
                for (size_t i = 0; i < nsols; i++)
                {
                opengv::essential_t E_tmp = Es_eigen[i] / Es_eigen[i].norm();
                E_tmp -= E_minSample;
                E_diffs[i] = E_tmp.norm();
                }
                int min_elem = std::distance(E_diffs.begin(), std::min_element(E_diffs.begin(), E_diffs.end()));*/

                std::vector<double> errSums(nsols, 0);
                std::vector<std::vector<double>> possible_models(nsols);
                for (size_t j = 0; j < nsols; j++)
                {
                    possible_models[j].resize(9);
                    for (size_t r = 0; r < 3; r++)
                        for (size_t c = 0; c < 3; c++)
                            possible_models[j][r * 3 + c] = Es_eigen[j](r, c);
                }
                for (unsigned int i = 0; i < usac_num_data_points_; ++i)
                {
                    if (usac_results_.inlier_flags_[i])
                    {
                        for (size_t j = 0; j < nsols; j++)
                        {
                            errSums[j] += PoseTools::getSampsonError(possible_models[j], input_points_denorm_, i);
                        }
                        if ((i > 3) && (i % 4 == 0))
                        {
                            std::vector<double> errSums_tmp = errSums;
                            std::partial_sort(errSums_tmp.begin(), errSums_tmp.begin() + 2, errSums_tmp.end());
                            if (errSums_tmp[0] < 0.66 * errSums_tmp[1])
                            {
                                break;
                            }
                        }
                    }
                }
                int min_elem = std::distance(errSums.begin(), std::min_element(errSums.begin(), errSums.end()));
                E_singele = Es_eigen[min_elem];
            }
            else if (nsols == 1)
                E_singele = Es_eigen[0];
            else
                return false;

            fivept_nister_essentials_denorm[0] = E_singele;
            fivept_nister_essentials[0] = m_T2_trans_e_inv * E_singele * m_T1_e_inv;
            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                {
                    *(models_[0] + r * 3 + c) = fivept_nister_essentials[0](r, c);
                    *(models_denorm_[0] + r * 3 + c) = fivept_nister_essentials_denorm[0](r, c);
                }
        }
        else if (refineMethod == USACConfig::REFINE_EIG_KNEIP || refineMethod == USACConfig::REFINE_EIG_KNEIP_WEIGHTS)
        {
            opengv::eigensolverOutput_t eig_out;
            opengv::essential_t E_eigen;
            cv::Mat R_tmp, t_tmp, E_tmp;
            //Variation of R as init for eigen-solver
            opengv::rotation_t R_init;
            if (used_estimator == USACConfig::ESTIM_EIG_KNEIP || weighted || refinedModelvalid)
            {
                if (weighted || refinedModelvalid)
                {
                    adapter_denorm->setR12(R_eigen_new);
                    eig_out.rotation = R_eigen_new;
                }
                else
                {
                    adapter_denorm->setR12(R_eigen);
                    eig_out.rotation = R_eigen;
                }
            }
            else
            {
                R_init = Eigen::Matrix3d::Identity();
                PoseTools::getPerturbedRotation(R_init, RAND_ROTATION_AMPLITUDE); //Check if the amplitude is too large or too small!
                adapter_denorm->setR12(R_init);
                eig_out.rotation = R_init;
            }

            if (refineMethod == USACConfig::REFINE_EIG_KNEIP_WEIGHTS && weighted)
            {
                opengv::bearingVectors_t bearingVectors1;
                opengv::bearingVectors_t bearingVectors2;
                std::vector<double> weights_vec;
                std::unique_ptr<opengv::relative_pose::CentralRelativeWeightingAdapter> adapter_denorm_weights; //Pointer to adapter for OpenGV if Kneip's Eigen solver with weights is used
                for (unsigned int i = 0; i < numPoints; i++)
                {
                    bearingVectors1.push_back(adapter_denorm->getBearingVector1(sample[i]));
                    bearingVectors2.push_back(adapter_denorm->getBearingVector2(sample[i]));
                    weights_vec.push_back((weights[i]));
                }
                adapter_denorm_weights.reset(new opengv::relative_pose::CentralRelativeWeightingAdapter(
                    bearingVectors1,
                    bearingVectors2,
                    weights_vec));

                adapter_denorm_weights->setR12(R_eigen_new);

                R_eigen_new = opengv::relative_pose::eigensolver(*adapter_denorm_weights, eig_out);
                t_eigen_new = eig_out.translation;
                t_eigen_new /= t_eigen_new.norm();
            }
            else
            {
                std::vector<int> indices;
                for (unsigned int i = 0; i < numPoints; ++i)
                {
                    indices.push_back((int)sample[i]);
                }
                R_eigen_new = opengv::relative_pose::eigensolver(*adapter_denorm, indices, eig_out);
                t_eigen_new = eig_out.translation;
                t_eigen_new /= t_eigen_new.norm();
                /*R_eigen_new.transposeInPlace();
                t_eigen_new = -1.0 * R_eigen_new * t_eigen_new;*/
            }
            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                    if (isnan(((long double) R_eigen_new(r, c))))
                    {
                        return false;
                    }
            if (!poselib::isMatRoationMat(R_eigen_new))
                return false;
            if (t_eigen_new.isZero(1e-3))
                return false;

            cv::eigen2cv(R_eigen_new, R_tmp);
            cv::eigen2cv(t_eigen_new, t_tmp);
            E_tmp = poselib::getEfromRT(R_tmp, t_tmp);
            cv::cv2eigen(E_tmp, E_eigen);
            fivept_nister_essentials_denorm[0] = E_eigen;
            fivept_nister_essentials[0] = m_T2_trans_e_inv * E_eigen * m_T1_e_inv;
            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                {
                    *(models_[0] + r * 3 + c) = fivept_nister_essentials[0](r, c);
                    *(models_denorm_[0] + r * 3 + c) = fivept_nister_essentials_denorm[0](r, c);
                }
        }
        else
        {
            std::cout << "Refinement algorithm not supported! Skipping!" << std::endl;
        }
    /*}
    else if (posetype == USACConfig::TRANS_ROTATION)
    {
        opengv::eigensolverOutput_t eig_out;
        adapter_denorm->setR12(R_eigen);

        std::vector<int> indices;
        for (unsigned int i = 0; i < numPoints; ++i)
        {
            indices.push_back((int)sample[i]);
        }

        R_eigen = opengv::relative_pose::eigensolver(*adapter_denorm, indices, eig_out);
        double sum_t = eig_out.translation.sum();
        if (sum_t > 1e-4)
        {
            opengv::essential_t E_eigen;
            cv::Mat R_tmp, t_tmp, E_tmp;
            std::cout << "It seems that the pose is not only a rotation! Changing to essential matrix." << std::endl;
            posetype = USACConfig::TRANS_ESSENTIAL;
            t_eigen = eig_out.translation;
            cv::eigen2cv(R_eigen, R_tmp);
            cv::eigen2cv(t_eigen, t_tmp);
            E_tmp = poselib::getEfromRT(R_tmp, t_tmp);
            cv::cv2eigen(E_tmp, E_eigen);
            fivept_nister_essentials_denorm[0] = E_eigen;
            fivept_nister_essentials[0] = m_T2_trans_e_inv * E_eigen * m_T1_e_inv;
            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                {
                    *(models_[0] + r * 3 + c) = fivept_nister_essentials[0](r, c);
                    *(models_denorm_[0] + r * 3 + c) = fivept_nister_essentials_denorm[0](r, c);
                }

        }
    }
    else if (posetype == USACConfig::TRANS_TRANSLATION)
    {
        opengv::eigensolverOutput_t eig_out;
        cv::Mat R_tmp;
        double roll, pitch, yaw;
        adapter_denorm->setR12(Eigen::Matrix3d::Identity());

        std::vector<int> indices;
        for (unsigned int i = 0; i < numPoints; ++i)
        {
            indices.push_back((int)sample[i]);
        }

        R_eigen = opengv::relative_pose::eigensolver(*adapter_denorm, indices, eig_out);
        t_eigen = eig_out.translation;

        cv::eigen2cv(R_eigen, R_tmp);
        poselib::getAnglesRotMat(R_tmp, roll, pitch, yaw, false);

        double sum_R = roll + pitch + yaw;
        if (sum_R > 0.01)
        {
            opengv::essential_t E_eigen;
            cv::Mat R_tmp, t_tmp, E_tmp;
            std::cout << "It seems that the pose is not only a translation! Changing to essential matrix." << std::endl;
            posetype = USACConfig::TRANS_ESSENTIAL;
            cv::eigen2cv(R_eigen, R_tmp);
            cv::eigen2cv(t_eigen, t_tmp);
            E_tmp = poselib::getEfromRT(R_tmp, t_tmp);
            cv::cv2eigen(E_tmp, E_eigen);
            fivept_nister_essentials_denorm[0] = E_eigen;
            fivept_nister_essentials[0] = m_T2_trans_e_inv * E_eigen * m_T1_e_inv;
            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                {
                    *(models_[0] + r * 3 + c) = fivept_nister_essentials[0](r, c);
                    *(models_denorm_[0] + r * 3 + c) = fivept_nister_essentials_denorm[0](r, c);
                }
        }
    }
    else if (posetype == USACConfig::TRANS_NO_MOTION)
    {
        R_eigen = Eigen::Matrix3d::Identity();
        t_eigen = Eigen::Vector3d::Zero();
    }
    else
    {
        std::cout << "Transformation type not supported!" << std::endl;
        return false;
    }*/

    return true;
}


// ============================================================================================
// validateSample: check if minimal sample is valid
// here, just returns true
// ============================================================================================
bool EssentialMatEstimator::validateSample()
{
    int j, k, i;

    for (i = 0; i < (int)usac_min_sample_size_; i++)
    {
        // check that the i-th selected point does not belong
        // to a line connecting some previously selected points
        for (j = 0; j < i; j++)
        {
            double pix = *(input_points_ + min_sample_[i] * 6) / *(input_points_ + min_sample_[i] * 6 + 2);
            double piy = *(input_points_ + min_sample_[i] * 6 + 1) / *(input_points_ + min_sample_[i] * 6 + 2);
            double pjx = *(input_points_ + min_sample_[j] * 6) / *(input_points_ + min_sample_[j] * 6 + 2);
            double pjy = *(input_points_ + min_sample_[j] * 6 + 1) / *(input_points_ + min_sample_[j] * 6 + 2);
            double dx1 = pjx - pix;
            double dy1 = pjy - piy;
            for (k = 0; k < j; k++)
            {
                double pkx = *(input_points_ + min_sample_[k] * 6) / *(input_points_ + min_sample_[k] * 6 + 2);
                double pky = *(input_points_ + min_sample_[k] * 6 + 1) / *(input_points_ + min_sample_[k] * 6 + 2);
                double dx2 = pkx - pix;
                double dy2 = pky - piy;
                if (fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                    break;
            }
            if (k < j)
                break;
        }
        if (j < i)
            break;
    }

    return i >= (int)usac_min_sample_size_ - 1;

    //return true; //was originally the only code in this function of USAC
}


// ============================================================================================
// validateModel: check if model computed from minimal sample is valid
// checks oriented constraints to determine model validity
// ============================================================================================
bool EssentialMatEstimator::validateModel(unsigned int modelIndex)//---------------------------------------> does this work with E?
{
    // check oriented constraints
    if (used_estimator != USACConfig::ESTIM_EIG_KNEIP)
    {
        double e[3], sig1, sig2;
        FTools::computeEpipole(e, models_[modelIndex]);

        sig1 = FTools::getOriSign(models_[modelIndex], e, input_points_ + 6 * min_sample_[0]);
        for (unsigned int i = 1; i < min_sample_.size(); ++i)
        {
            sig2 = FTools::getOriSign(models_[modelIndex], e, input_points_ + 6 * min_sample_[i]);
            if (sig1 * sig2 < 0)
            {
                return false;
            }
        }
    }
    return true;
}


// ============================================================================================
// evaluateModel: test model against all/subset of the data points
// ============================================================================================
bool EssentialMatEstimator::evaluateModel(unsigned int modelIndex, unsigned int* numInliers, unsigned int* numPointsTested)
{
    double rx, ry, rwc, ryc, rxc, r, temp_err;
    double* model = models_denorm_[modelIndex];
    double* pt;
    auto current_err_array = err_ptr_[0];
    bool good_flag = true;
    double lambdaj, lambdaj_1 = 1.0;
    *numInliers = 0;
    *numPointsTested = 0;
    unsigned int pt_index;

    for (unsigned int i = 0; i < usac_num_data_points_; ++i)
    {
        // get index of point to be verified
        if (eval_pool_index_ > usac_num_data_points_ - 1)
        {
            eval_pool_index_ = 0;
        }
        pt_index = evaluation_pool_[eval_pool_index_];
        ++eval_pool_index_;

        // compute sampson error
        pt = input_points_denorm_ + 6 * pt_index;
        rxc = (*model) * (*(pt + 3)) + (*(model + 3)) * (*(pt + 4)) + (*(model + 6));
        ryc = (*(model + 1)) * (*(pt + 3)) + (*(model + 4)) * (*(pt + 4)) + (*(model + 7));
        rwc = (*(model + 2)) * (*(pt + 3)) + (*(model + 5)) * (*(pt + 4)) + (*(model + 8));
        r = ((*(pt)) * rxc + (*(pt + 1)) * ryc + rwc);
        rx = (*model) * (*(pt)) + (*(model + 1)) * (*(pt + 1)) + (*(model + 2));
        ry = (*(model + 3)) * (*(pt)) + (*(model + 4)) * (*(pt + 1)) + (*(model + 5));
        temp_err = r*r / (rxc*rxc + ryc*ryc + rx*rx + ry*ry);
        *(current_err_array + pt_index) = temp_err;

        if (temp_err < usac_inlier_threshold_)
        {
            ++(*numInliers);
        }

        if (usac_verif_method_ == USACConfig::VERIF_SPRT)
        {
            if (temp_err < usac_inlier_threshold_)
            {
                lambdaj = lambdaj_1 * (sprt_delta_ / sprt_epsilon_);
            }
            else
            {
                lambdaj = lambdaj_1 * ((1 - sprt_delta_) / (1 - sprt_epsilon_));
            }
            if (lambdaj <= DBL_EPSILON) lambdaj = DBL_EPSILON * 10;

            if (lambdaj > decision_threshold_sprt_)
            {
                good_flag = false;
                *numPointsTested = i + 1;
                return good_flag;
            }
            else
            {
                lambdaj_1 = lambdaj;
            }
        }
    }
    *numPointsTested = usac_num_data_points_;
    return good_flag;
}

// ============================================================================================
// evaluateModelRot: test model against all/subset of the data points
// ============================================================================================
bool EssentialMatEstimator::evaluateModelRot(const opengv::rotation_t &model,
        unsigned int* numInliers,
        unsigned int* numPointsTested)
{
    double temp_err;
    auto current_err_array = err_ptr_[0];
    bool good_flag = true;
    double lambdaj, lambdaj_1 = 1.0;
    *numInliers = 0;
    *numPointsTested = 0;
    unsigned int pt_index;

    for (unsigned int i = 0; i < usac_num_data_points_; ++i)
    {
        // get index of point to be verified
        if (eval_pool_index_ > usac_num_data_points_ - 1)
        {
            eval_pool_index_ = 0;
        }
        pt_index = evaluation_pool_[eval_pool_index_];
        ++eval_pool_index_;

        opengv::bearingVector_t f1 = adapter_denorm->getBearingVector1(pt_index);
        opengv::bearingVector_t f2 = adapter_denorm->getBearingVector2(pt_index);

        //unrotate bearing-vector f2
        opengv::bearingVector_t f2_unrotated = model * f2;

        //bearing-vector based outlier criterium (select threshold accordingly):
        //1-(f1'*f2) = 1-cos(alpha) \in [0:2]
        temp_err = 1.0 - (f1.transpose() * f2_unrotated);
        *(current_err_array + pt_index) = temp_err;

        if (temp_err < poseDegenTheshold)
        {
            ++(*numInliers);
        }

        if (usac_verif_method_ == USACConfig::VERIF_SPRT)
        {
            if (temp_err < poseDegenTheshold)
            {
                lambdaj = lambdaj_1 * (sprt_delta_ / sprt_epsilon_);
            }
            else
            {
                lambdaj = lambdaj_1 * ((1 - sprt_delta_) / (1 - sprt_epsilon_));
            }
            if (lambdaj <= DBL_EPSILON) lambdaj = DBL_EPSILON * 10;

            if (lambdaj > decision_threshold_sprt_)
            {
                good_flag = false;
                *numPointsTested = i + 1;
                return good_flag;
            }
            else
            {
                lambdaj_1 = lambdaj;
            }
        }
    }
    *numPointsTested = usac_num_data_points_;
    return good_flag;
}

// ============================================================================================
// evaluateModelTrans: test model against all/subset of the data points
// ============================================================================================
bool EssentialMatEstimator::evaluateModelTrans(const opengv::translation_t &model, unsigned int* numInliers, unsigned int* numPointsTested)
{
    double temp_err;
    auto current_err_array = err_ptr_[0];
    bool good_flag = true;
    double lambdaj, lambdaj_1 = 1.0;
    *numInliers = 0;
    *numPointsTested = 0;
    unsigned int pt_index;

    //translation_t translation = model.col(3);
    opengv::rotation_t rotation = opengv::rotation_t::Identity();
    adapter_denorm->sett12(model);
    adapter_denorm->setR12(rotation);

    opengv::transformation_t inverseSolution;
    inverseSolution.block<3, 3>(0, 0) = rotation.transpose();
    inverseSolution.col(3) = -inverseSolution.block<3, 3>(0, 0)*model;

    Eigen::Matrix<double, 4, 1> p_hom;
    p_hom[3] = 1.0;

    for (unsigned int i = 0; i < usac_num_data_points_; ++i)
    {
        // get index of point to be verified
        if (eval_pool_index_ > usac_num_data_points_ - 1)
        {
            eval_pool_index_ = 0;
        }
        pt_index = evaluation_pool_[eval_pool_index_];
        ++eval_pool_index_;

        p_hom.block<3, 1>(0, 0) =
            opengv::triangulation::triangulate2(*adapter_denorm, pt_index);
        opengv::bearingVector_t reprojection1 = p_hom.block<3, 1>(0, 0);
        opengv::bearingVector_t reprojection2 = inverseSolution * p_hom;
        reprojection1 = reprojection1 / reprojection1.norm();
        reprojection2 = reprojection2 / reprojection2.norm();
        opengv::bearingVector_t f1 = adapter_denorm->getBearingVector1(pt_index);
        opengv::bearingVector_t f2 = adapter_denorm->getBearingVector2(pt_index);

        //bearing-vector based outlier criterium (select threshold accordingly):
        //1-(f1'*f2) = 1-cos(alpha) \in [0:2]
        double reprojError1 = 1.0 - (f1.transpose() * reprojection1);
        double reprojError2 = 1.0 - (f2.transpose() * reprojection2);
        temp_err = reprojError1 + reprojError2;
        *(current_err_array + pt_index) = temp_err;

        if (temp_err < poseDegenTheshold)
        {
            ++(*numInliers);
        }

        if (usac_verif_method_ == USACConfig::VERIF_SPRT)
        {
            if (temp_err < poseDegenTheshold)
            {
                lambdaj = lambdaj_1 * (sprt_delta_ / sprt_epsilon_);
            }
            else
            {
                lambdaj = lambdaj_1 * ((1 - sprt_delta_) / (1 - sprt_epsilon_));
            }
            if (lambdaj <= DBL_EPSILON) lambdaj = DBL_EPSILON * 10;

            if (lambdaj > decision_threshold_sprt_)
            {
                good_flag = false;
                *numPointsTested = i + 1;
                return good_flag;
            }
            else
            {
                lambdaj_1 = lambdaj;
            }
        }
    }
    *numPointsTested = usac_num_data_points_;
    return good_flag;
}


// ============================================================================================
// testSolutionDegeneracy: check if model is degenerate
// test if >=5 points in the sample are on a plane, or if the pose is only a rotation, translation, or no motion
// ============================================================================================
void EssentialMatEstimator::testSolutionDegeneracy(bool* degenerateModel, bool* upgradeModel)
{
    *degenerateModel = false;
    *upgradeModel = false;
    degeneracyType = DEGEN_NOT_FOUND;

    //Degeneracy induced by a plane (homography):
    if(enableHDegen)
        testSolutionDegeneracyH(degenerateModel, upgradeModel);

    if (upgradeModel || !enableHDegen)
    {
        degeneracyType = DEGEN_H;
        //Degeneracy due to rotaion only:
        testSolutionDegeneracyRot(degenerateModel);

        if (!enableHDegen && (degeneracyType & DEGEN_UPGRADE))
            *upgradeModel = true;

        //Degeneracy due to translation only:
        /*testSolutionDegeneracyTrans(degenerateModel);

        if (!enableHDegen && (degeneracyType & DEGEN_UPGRADE))
            *upgradeModel = true;*/

        if (degeneracyType == (DEGEN_ROT_TRANS | DEGEN_UPGRADE))
        {
            //Degeneracy due to no motion:
            testSolutionDegeneracyNoMot(degenerateModel);
        }
    }
}

// ============================================================================================
// testSolutionDegeneracy: check if model is degenerate
// test if >=5 points in the sample are on a plane
// ============================================================================================
void EssentialMatEstimator::testSolutionDegeneracyH(bool* degenerateModel, bool* upgradeModel)
{
    // make up the tuples to be used to check for degeneracy
    unsigned int degen_sample_indices[] = { 0, 1, 2, 3,
        1, 2, 3, 4,
        0, 2, 3, 4,
        0, 1, 3, 4,
        0, 1, 2, 4 };

    // the above tuples need to be tested on the remaining points for each case
    unsigned int test_point_indices[] = { 4,
        0,
        1,
        2,
        3 };

    unsigned int *sample_pos = degen_sample_indices;
    unsigned int *test_pos = test_point_indices;
    double h[9];
    double T2_inv[9], T2_H[9];
    for (unsigned int i = 0; i < 9; ++i)
    {
        T2_inv[i] = m_T2_[i];
    }
    MathTools::minv(T2_inv, 3);

    std::vector<unsigned int> sample(5), test(1);
    std::vector<double> errs;
    for (unsigned int i = 0; i < 5; ++i)
    {
        // compute H from the current set of 4 points
        for (unsigned int j = 0; j < 4; ++j)
        {
            sample[j] = min_sample_[sample_pos[j]];
        }
        FTools::computeHFromMinCorrs(sample, 4, usac_num_data_points_, degen_data_matrix_, h);
        MathTools::mmul(T2_H, T2_inv, h, 3);
        MathTools::mmul(h, T2_H, m_T1_, 3);

        // check test points to see how many are consistent
        for (unsigned int j = 0; j < 1; ++j)
        {
            test[j] = min_sample_[test_pos[j]];
        }
        unsigned int num_inliers = FTools::getHError(test, 1, errs, input_points_denorm_, h, degen_homog_threshold_);
        for (unsigned int j = 0, count = 4; j < 1; ++j)
        {
            if (errs[j] < degen_homog_threshold_)
            {
                sample[count++] = test[j];
            }
        }

        // if at least 1 inlier in the test points, then h-degenerate sample found
        if (num_inliers > 0)
        {
            // find inliers from all data points
            num_inliers = FTools::getHError(evaluation_pool_, usac_num_data_points_, errs, input_points_denorm_, h, degen_homog_threshold_);
            //std::cout << "Degenerate sample found with " << num_inliers << " inliers" << std::endl;

            // refine with least squares fit
            unsigned int count = 0;
            std::vector<unsigned int> inlier_sample(num_inliers);
            for (unsigned int j = 0; j < usac_num_data_points_; ++j)
            {
                if (errs[j] < degen_homog_threshold_)
                {
                    inlier_sample[count++] = evaluation_pool_[j];
                }
            }
            FTools::computeHFromCorrs(inlier_sample, inlier_sample.size(), usac_num_data_points_, degen_data_matrix_, h);
            MathTools::mmul(T2_H, T2_inv, h, 3);
            MathTools::mmul(h, T2_H, m_T1_, 3);

            // find support of homography
            num_inliers = FTools::getHError(evaluation_pool_, usac_num_data_points_, errs, input_points_denorm_, h, degen_homog_threshold_);
            //std::cout << "Refined model has " << num_inliers << " inliers" << std::endl;
#if 1
            if (num_inliers < usac_results_.best_inlier_count_ / 5)
            {
                sample_pos += 4;
                test_pos += 1;
                continue;
            }
#endif
            // set flag
            *degenerateModel = true;

            // if largest degenerate model found so far, store results
            if (num_inliers > usac_results_.degen_inlier_count_)
            {
                // set flag
                *upgradeModel = true;

                // refine with final least squares fit
                count = 0;
                inlier_sample.resize(num_inliers);
                for (unsigned int j = 0; j < usac_num_data_points_; ++j)
                {
                    if (errs[j] < degen_homog_threshold_)
                    {
                        inlier_sample[count++] = evaluation_pool_[j];
                    }
                }
                FTools::computeHFromCorrs(inlier_sample, inlier_sample.size(), usac_num_data_points_, degen_data_matrix_, h);

                usac_results_.degen_inlier_count_ = num_inliers;
                // store homography
                for (unsigned int j = 0; j < 9; ++j)
                {
                    degen_final_model_params_[j] = h[j];
                }
                // store inliers and outliers - for use in model completion
                for (unsigned int j = 0; j < usac_num_data_points_; ++j)
                {
                    if (errs[j] < degen_homog_threshold_)
                    {
                        usac_results_.degen_inlier_flags_[evaluation_pool_[j]] = 1;
                        degen_outlier_flags_[evaluation_pool_[j]] = 0;
                    }
                    else
                    {
                        degen_outlier_flags_[evaluation_pool_[j]] = 1;
                        usac_results_.degen_inlier_flags_[evaluation_pool_[j]] = 0;
                    }
                }
                // store the degenerate points from the minimal sample
                usac_results_.degen_sample_ = sample;

            } // end store denerate results
        } // end check for one model degeneracy

        sample_pos += 4;
        test_pos += 1;

    } // end check for all combinations in the minimal sample
}

// ============================================================================================
// testSolutionDegeneracy: check if model is degenerate
// test if >=3 point correspondences in the sample are best described using a rotation
// ============================================================================================
void EssentialMatEstimator::testSolutionDegeneracyRot(bool* degenerateModel)
{
    // make up the tuples to be used to check for degeneracy
    unsigned int degen_sample_indices[] = { 0, 1,
        0, 2,
        0, 3,
        0, 4,
        1, 2,
        1, 3,
        1, 4,
        2, 3,
        2, 4,
        3, 4 };

    // the above tuples need to be tested on the remaining points for each case
    unsigned int test_point_indices[] = { 2, 3, 4,
        1, 3, 4,
        1, 2, 4,
        1, 2, 3,
        0, 3, 4,
        0, 2, 4,
        0, 2, 3,
        0, 1, 4,
        0, 1, 3,
        0, 1, 2 };

    unsigned int *sample_pos = degen_sample_indices;
    unsigned int *test_pos = test_point_indices;
    opengv::rotation_t R_;

    std::vector<int> sample(5);
    std::vector<unsigned int> test(3);
    std::vector<double> errs;
    for (unsigned int i = 0; i < 10; ++i)
    {
        // compute H from the current set of 4 points
        for (unsigned int j = 0; j < 2; ++j)
        {
            sample[j] = (int)min_sample_[sample_pos[j]];
        }
        R_ = opengv::relative_pose::twopt_rotationOnly(*adapter_denorm, sample);

        // check test points to see how many are consistent
        for (unsigned int j = 0; j < 3; ++j)
        {
            test[j] = min_sample_[test_pos[j]];
        }
        unsigned int num_inliers = PoseTools::getRotError(test, 3, errs, adapter_denorm, R_, poseDegenTheshold);

        unsigned int count1 = 2;
        for (unsigned int j = 0; j < 3; ++j)
        {
            if (errs[j] < poseDegenTheshold)
            {
                sample[count1++] = test[j];
            }
        }

        // if at least 1 inlier in the test points, then h-degenerate sample found
        if (num_inliers > 0)
        {
            // find inliers from all data points
            num_inliers = PoseTools::getRotError(evaluation_pool_, usac_num_data_points_, errs, adapter_denorm, R_, poseDegenTheshold);
            if (num_inliers < 2)
            {
                sample_pos += 2;
                test_pos += 3;
                continue;
            }
            //std::cout << "Degenerate sample found with " << num_inliers << " inliers" << std::endl;

            // refine with least squares fit
            unsigned int count = 0;
            std::vector<int> inlier_sample(num_inliers);
            for (unsigned int j = 0; j < usac_num_data_points_; ++j)
            {
                if (errs[j] < poseDegenTheshold)
                {
                    inlier_sample[count++] = (int)evaluation_pool_[j];
                    if(count1 < 5)
                        sample[count1++] = (int)evaluation_pool_[j];
                }
            }
            R_ = opengv::relative_pose::rotationOnly(*adapter_denorm, inlier_sample);

            // find support of rotation
            num_inliers = PoseTools::getRotError(evaluation_pool_, usac_num_data_points_, errs, adapter_denorm, R_, poseDegenTheshold);
            //std::cout << "Refined model has " << num_inliers << " inliers" << std::endl;
#if 1
            if (num_inliers < usac_results_.best_inlier_count_ / 5)
            {
                sample_pos += 2;
                test_pos += 3;
                continue;
            }
#endif
            // set flag
            *degenerateModel = true;
            if(degeneracyType != (degeneracyType & (DEGEN_UPGRADE | DEGEN_ROT_TRANS)))
                degeneracyType = DEGEN_ROT_TRANS;

            // if largest degenerate model found so far, store results
            if (num_inliers > usac_results_.degen_inlier_count_rot)
            {
                // set flag
                degeneracyType |= DEGEN_UPGRADE;
                //*upgradeModel = true;

                // refine with final least squares fit
                count = 0;
                inlier_sample.resize(num_inliers);
                for (unsigned int j = 0; j < usac_num_data_points_; ++j)
                {
                    if (errs[j] < poseDegenTheshold)
                    {
                        inlier_sample[count++] = evaluation_pool_[j];
                    }
                }
                R_ = opengv::relative_pose::rotationOnly(*adapter_denorm, inlier_sample);

                usac_results_.degen_inlier_count_rot = num_inliers;
                // store rotation
                for (unsigned int r = 0; r < 3; ++r)
                    for (unsigned int c = 0; c < 3; ++c)
                        degen_final_model_params_rot[r * 3 + c] = R_(r, c);
                R_eigen_degen = R_;

                // store inliers and outliers - for use in model completion
                for (unsigned int j = 0; j < usac_num_data_points_; ++j)
                {
                    if (errs[j] < poseDegenTheshold)
                    {
                        usac_results_.degen_inlier_flags_rot[evaluation_pool_[j]] = 1;
                        degen_outlier_flags_rot[evaluation_pool_[j]] = 0;
                    }
                    else
                    {
                        degen_outlier_flags_rot[evaluation_pool_[j]] = 1;
                        usac_results_.degen_inlier_flags_rot[evaluation_pool_[j]] = 0;
                    }
                }
                // store the degenerate points from the minimal sample
                degen_sample_rot = sample;

            } // end store denerate results
        } // end check for one model degeneracy

        sample_pos += 2;
        test_pos += 3;

    } // end check for all combinations in the minimal sample
}

// ============================================================================================
// testSolutionDegeneracy: check if model is degenerate
// test if >=3 point correspondences in the sample are best described using a translation
// ============================================================================================
void EssentialMatEstimator::testSolutionDegeneracyTrans(bool* degenerateModel)
{
    // make up the tuples to be used to check for degeneracy
    unsigned int degen_sample_indices[] = { 0, 1,
        0, 2,
        0, 3,
        0, 4,
        1, 2,
        1, 3,
        1, 4,
        2, 3,
        2, 4,
        3, 4 };

    // the above tuples need to be tested on the remaining points for each case
    unsigned int test_point_indices[] = { 2, 3, 4,
        1, 3, 4,
        1, 2, 4,
        1, 2, 3,
        0, 3, 4,
        0, 2, 4,
        0, 2, 3,
        0, 1, 4,
        0, 1, 3,
        0, 1, 2 };

    unsigned int *sample_pos = degen_sample_indices;
    unsigned int *test_pos = test_point_indices;
    opengv::translation_t t_;
    opengv::rotation_t R_ = opengv::rotation_t::Identity();
    adapter_denorm->setR12(R_);

    std::vector<int> sample(5);
    std::vector<unsigned int> test(3);
    std::vector<double> errs;
    for (unsigned int i = 0; i < 10; ++i)
    {
        // compute H from the current set of 4 points
        for (unsigned int j = 0; j < 2; ++j)
        {
            sample[j] = (int)min_sample_[sample_pos[j]];
        }
        t_ = opengv::relative_pose::twopt(*adapter_denorm, false, sample);

        // check test points to see how many are consistent
        for (unsigned int j = 0; j < 3; ++j)
        {
            test[j] = min_sample_[test_pos[j]];
        }
        unsigned int num_inliers = PoseTools::getTransError(test, 3, errs, adapter_denorm, t_, poseDegenTheshold);

        for (unsigned int j = 0, count = 2; j < 3; ++j)
        {
            if (errs[j] < poseDegenTheshold)
            {
                sample[count++] = test[j];
            }
        }

        // if at least 1 inlier in the test points, then t-degenerate sample found
        if (num_inliers > 0)
        {
            opengv::eigensolverOutput_t eig_out;
            cv::Mat R_tmp;
            double roll, pitch, yaw;
            // find inliers from all data points
            num_inliers = PoseTools::getTransError(evaluation_pool_, usac_num_data_points_, errs, adapter_denorm, t_, poseDegenTheshold);
            if (num_inliers < 2)
            {
                sample_pos += 2;
                test_pos += 3;
                continue;
            }
            //std::cout << "Degenerate sample found with " << num_inliers << " inliers" << std::endl;

            // refine with least squares fit
            unsigned int count = 0;
            std::vector<int> inlier_sample(num_inliers);
            for (unsigned int j = 0; j < usac_num_data_points_; ++j)
            {
                if (errs[j] < poseDegenTheshold)
                {
                    inlier_sample[count++] = (int)evaluation_pool_[j];
                }
            }
            eig_out.rotation = R_;
            R_ = opengv::relative_pose::eigensolver(*adapter_denorm, inlier_sample, eig_out);
            t_ = eig_out.translation;
            t_ /= t_.norm();
            cv::eigen2cv(R_, R_tmp);
            poselib::getAnglesRotMat(R_tmp, roll, pitch, yaw, false);
            double sum_R = roll + pitch + yaw;
            if (sum_R > 0.01)
                continue;

            // find support of translation
            num_inliers = PoseTools::getTransError(evaluation_pool_, usac_num_data_points_, errs, adapter_denorm, t_, poseDegenTheshold);
            //std::cout << "Refined model has " << num_inliers << " inliers" << std::endl;
#if 1
            if (num_inliers < usac_results_.best_inlier_count_ / 5)
            {
                sample_pos += 2;
                test_pos += 3;
                continue;
            }
#endif
            // set flag
            *degenerateModel = true;
            degeneracyType = DEGEN_ROT_TRANS;

            // if largest degenerate model found so far, store results
            if (num_inliers > usac_results_.degen_inlier_count_trans)
            {
                // set flag
                //*upgradeModel = true;
                degeneracyType |= DEGEN_UPGRADE;

                // refine with final least squares fit
                count = 0;
                inlier_sample.resize(num_inliers);
                for (unsigned int j = 0; j < usac_num_data_points_; ++j)
                {
                    if (errs[j] < poseDegenTheshold)
                    {
                        inlier_sample[count++] = evaluation_pool_[j];
                    }
                }
                eig_out.rotation = R_;
                R_ = opengv::relative_pose::eigensolver(*adapter_denorm, inlier_sample, eig_out);
                t_ = eig_out.translation;
                t_ /= t_.norm();
                cv::eigen2cv(R_, R_tmp);
                poselib::getAnglesRotMat(R_tmp, roll, pitch, yaw, false);
                sum_R = roll + pitch + yaw;
                if (sum_R > 0.01)
                    continue;

                usac_results_.degen_inlier_count_trans = num_inliers;
                // store translation
                for (unsigned int r = 0; r < 3; ++r)
                    degen_final_model_params_trans[r] = t_(r);

                // store inliers and outliers - for use in model completion
                for (unsigned int j = 0; j < usac_num_data_points_; ++j)
                {
                    if (errs[j] < poseDegenTheshold)
                    {
                        usac_results_.degen_inlier_flags_trans[evaluation_pool_[j]] = 1;
                        degen_outlier_flags_trans[evaluation_pool_[j]] = 0;
                    }
                    else
                    {
                        degen_outlier_flags_trans[evaluation_pool_[j]] = 1;
                        usac_results_.degen_inlier_flags_trans[evaluation_pool_[j]] = 0;
                    }
                }
                // store the degenerate points from the minimal sample
                degen_sample_trans = sample;

            } // end store denerate results
        } // end check for one model degeneracy

        sample_pos += 2;
        test_pos += 3;

    } // end check for all combinations in the minimal sample
}

// ============================================================================================
// testSolutionDegeneracy: check if model is degenerate
// test if >=3 point correspondences in the sample are best described by no motion
// ============================================================================================
void EssentialMatEstimator::testSolutionDegeneracyNoMot(bool* degenerateModel)
{
    if (usac_results_.degen_inlier_count_noMot > 0) //As there are no estimated model parameters, calculating the inliers once is sufficient
        return;
    // the above tuples need to be tested on the remaining points for each case
    unsigned int test_point_indices[] = { 0, 1, 2, 3, 4};

    std::vector<unsigned int> test(5);
    std::vector<double> errs;

    // check test points to see how many are consistent
    for (unsigned int j = 0; j < 5; ++j)
    {
        test[j] = min_sample_[test_point_indices[j]];
    }
    unsigned int num_inliers = PoseTools::getNoMotError(test, 5, errs, adapter_denorm, poseDegenTheshold);

    degen_sample_noMot.clear();
    for (unsigned int j = 0; j < 5; ++j)
    {
        if (errs[j] < poseDegenTheshold)
        {
            degen_sample_noMot.push_back((int)test[j]);
        }
    }

    // if at least 1 inlier in the test points, then h-degenerate sample found
    if (num_inliers > 0)
    {
        // find inliers from all data points
        num_inliers = PoseTools::getNoMotError(evaluation_pool_, usac_num_data_points_, errs, adapter_denorm, poseDegenTheshold);

#if 1
        if (num_inliers < usac_results_.best_inlier_count_ / 5)
        {
            return;
        }
#endif
        // set flag
        *degenerateModel = true;
        if (((double)num_inliers > 0.7 * (double)usac_results_.degen_inlier_count_rot))// || ((double)num_inliers > 0.7 * (double)usac_results_.degen_inlier_count_trans))
            if (degeneracyType != (degeneracyType & (DEGEN_UPGRADE | DEGEN_NO_MOT)))
                degeneracyType = DEGEN_NO_MOT;

        // if largest degenerate model found so far, store results
        if (num_inliers > usac_results_.degen_inlier_count_noMot)
        {
            if (((double)num_inliers > 0.7 * (double)usac_results_.degen_inlier_count_rot))// || ((double)num_inliers > 0.7 * (double)usac_results_.degen_inlier_count_trans))
                degeneracyType |= DEGEN_UPGRADE;
            usac_results_.degen_inlier_count_noMot = num_inliers;

            // store inliers and outliers - for use in model completion
            for (unsigned int j = 0; j < usac_num_data_points_; ++j)
            {
                if (errs[j] < poseDegenTheshold)
                {
                    usac_results_.degen_inlier_flags_noMot[evaluation_pool_[j]] = 1;
                    degen_outlier_flags_noMot[evaluation_pool_[j]] = 0;
                    if(degen_sample_noMot.size() < 5)
                        degen_sample_noMot.push_back((int)evaluation_pool_[j]);
                }
                else
                {
                    degen_outlier_flags_noMot[evaluation_pool_[j]] = 1;
                    usac_results_.degen_inlier_flags_noMot[evaluation_pool_[j]] = 0;
                }
            }
        } // end store denerate results
    } // end check for one model degeneracy
}

// ============================================================================================
// upgradeDegenerateModel: try to upgrade degenerate model to non-degenerate by sampling from
// the set of outliers to the degenerate model
// ============================================================================================
unsigned int EssentialMatEstimator::upgradeDegenerateModel()
{
    unsigned int best_upgrade_inliers = usac_results_.best_inlier_count_;
    unsigned int best_upgrade_inliers_rot = usac_results_.degen_inlier_count_rot;
    unsigned int best_upgrade_inliers_trans = usac_results_.degen_inlier_count_trans;
    unsigned int num_outliers = usac_num_data_points_ - usac_results_.degen_inlier_count_;

    if (num_outliers < 2) {
        return 0;
    }

    //The test for H degeneracy is not important - so it should remain disabled
    if (enableHDegen && !(degeneracyType & DEGEN_UPGRADE))
    {
        std::vector<unsigned int> outlier_indices(num_outliers);
        unsigned int count = 0;
        for (unsigned int i = 0; i < usac_num_data_points_; ++i)
        {
            if (degen_outlier_flags_[i])
            {
                outlier_indices[count++] = i;
            }
        }
        std::vector<unsigned int> outlier_sample(2);
        //std::fill(errs_.begin(), errs_.begin() + usac_num_data_points_, DBL_MAX);
        std::fill(err_ptr_[0], err_ptr_[0] + usac_num_data_points_, DBL_MAX);
        auto current_err_array = err_ptr_[0];

        double* pt1_index, *pt2_index;
        double x1[3], x1p[3], x2[3], x2p[3];
        double temp[3], l1[3], l2[3], ep[3];
        double skew_sym_ep[9];
        double T2_F[9];

        if (ransacLikeUpgradeDegenerate)
        {
            opengv::eigensolverOutput_t eig_out;
            opengv::essential_t E_eigen;
            opengv::translation_t t_eigen_tmp;
            opengv::rotation_t R_eigen_tmp;
            cv::Mat R_tmp, t_tmp, E_tmp;
            for (unsigned int i = 0; i < degen_max_upgrade_samples_; ++i)
            {
                generateUniformRandomSample(num_outliers, 1, &outlier_sample);
                std::vector<int> index(5);
                for (unsigned int j = 0; j < 4; j++)
                {
                    index[j] = (int)usac_results_.degen_sample_[j];
                }
                index[5] = (int)outlier_indices[outlier_sample[0]];

                //Variation of R as init for eigen-solver
                opengv::rotation_t R_init = Eigen::Matrix3d::Identity();
                PoseTools::getPerturbedRotation(R_init, RAND_ROTATION_AMPLITUDE); //Check if the amplitude is too large or too small!

                adapter_denorm->setR12(R_init);
                R_eigen_tmp = opengv::relative_pose::eigensolver(*adapter_denorm, index);
                t_eigen_tmp = eig_out.translation;
                cv::eigen2cv(R_eigen_tmp, R_tmp);
                cv::eigen2cv(t_eigen_tmp, t_tmp);
                E_tmp = poselib::getEfromRT(R_tmp, t_tmp);
                cv::cv2eigen(E_tmp, E_eigen);
                for (size_t r = 0; r < 3; r++)
                    for (size_t c = 0; c < 3; c++)
                        *(models_denorm_[0] + r * 3 + c) = E_eigen(r, c);

                unsigned int num_inliers, num_pts_tested;
                evaluateModel(0, &num_inliers, &num_pts_tested);

                if (num_inliers > best_upgrade_inliers)
                {
                    for (unsigned int j = 0; j < 5; j++)
                        usac_results_.degen_sample_[j] = (unsigned int)index[j];

                    min_sample_ = usac_results_.degen_sample_;
                    storeSolution(0, num_inliers);
                    best_upgrade_inliers = num_inliers;

                    unsigned int diffToNumOutliers = 0;
                    count = 0;
                    for (auto &j : outlier_indices)
                    {
                        if (*(current_err_array + j) < usac_inlier_threshold_)
                        {
                            ++count;
                        }
                        else if (std::round(*(current_err_array + j) - DBL_MAX) == 0)
                        {
                            ++diffToNumOutliers;
                        }
                    }
                    unsigned int num_samples = updateStandardStopping(count, num_outliers - diffToNumOutliers, 1);
                    if (num_samples < degen_max_upgrade_samples_)
                    {
                        degen_max_upgrade_samples_ = num_samples;
                    }
                }
            }
        }
        else
        {
            unsigned int cntNoEMat = 0;
            for (unsigned int i = 0; i < degen_max_upgrade_samples_; ++i)
            {
                generateUniformRandomSample(num_outliers, 2, &outlier_sample);

                pt1_index = input_points_ + 6 * outlier_indices[outlier_sample[0]];
                pt2_index = input_points_ + 6 * outlier_indices[outlier_sample[1]];

                x1[0] = pt1_index[0]; x1[1] = pt1_index[1]; x1[2] = 1.0;
                x1p[0] = pt1_index[3]; x1p[1] = pt1_index[4]; x1p[2] = 1.0;
                x2[0] = pt2_index[0]; x2[1] = pt2_index[1]; x2[2] = 1.0;
                x2p[0] = pt2_index[3]; x2p[1] = pt2_index[4]; x2p[2] = 1.0;

                //-------------------------------> Does this work for an Essential matrix????
                MathTools::vmul(temp, &degen_final_model_params_[0], x1, 3);
                MathTools::crossprod(l1, temp, x1p, 1);

                MathTools::vmul(temp, &degen_final_model_params_[0], x2, 3);
                MathTools::crossprod(l2, temp, x2p, 1);

                MathTools::crossprod(ep, l1, l2, 1);

                MathTools::skew_sym(skew_sym_ep, ep);
                MathTools::mmul(models_[0], skew_sym_ep, &degen_final_model_params_[0], 3);
                MathTools::mmul(T2_F, m_T2_trans_, models_[0], 3);
                MathTools::mmul(models_denorm_[0], T2_F, m_T1_, 3);

                //Check if the 2 eigenvalues are approximately equal
                Eigen::Matrix3d V, E;
                E << *(models_denorm_[0]), *(models_denorm_[0] + 1), *(models_denorm_[0] + 2),
                    *(models_denorm_[0] + 3), *(models_denorm_[0] + 4), *(models_denorm_[0] + 5),
                    *(models_denorm_[0] + 6), *(models_denorm_[0] + 7), *(models_denorm_[0] + 8);
                Eigen::JacobiSVD<Eigen::Matrix3d > svdE(E, Eigen::ComputeFullV);

                if (std::abs(1.0 - svdE.singularValues()(0) / svdE.singularValues()(1)) < 0.1)
                {
                    unsigned int num_inliers, num_pts_tested;
                    evaluateModel(0, &num_inliers, &num_pts_tested);

                    if (num_inliers > best_upgrade_inliers)
                    {
                        usac_results_.degen_sample_[3] = outlier_indices[outlier_sample[0]];//Overwrite samples on the plane with samples off the plane (for F (7pt alg), this samples are added)
                        usac_results_.degen_sample_[4] = outlier_indices[outlier_sample[1]];//Overwrite samples on the plane with samples off the plane
                        min_sample_ = usac_results_.degen_sample_;
                        storeSolution(0, num_inliers);
                        best_upgrade_inliers = num_inliers;

                        unsigned int diffToNumOutliers = 0;
                        count = 0;
                        for (auto &j : outlier_indices)
                        {
                            if (*(current_err_array + j) < usac_inlier_threshold_)
                            {
                                ++count;
                            }
                            else if (std::round(*(current_err_array + j) - DBL_MAX) == 0)
                            {
                                ++diffToNumOutliers;
                            }
                        }
                        unsigned int num_samples = updateStandardStopping(count, num_outliers - diffToNumOutliers, 2);
                        //std::cout << "Inliers = " << num_inliers << ", in/out = " << count << "/" << num_outliers
                        //	      << ". Num samples = " << num_samples << std::endl;
                        if (num_samples < degen_max_upgrade_samples_)
                        {
                            degen_max_upgrade_samples_ = num_samples;
                        }

                    }
                    cntNoEMat++;
                }
                if ((double)cntNoEMat / (double)i < 0.1)
                {
                    std::cout << "Upgrading E like F from H does not work!" << std::endl;
                    return best_upgrade_inliers;
                }
            }
        }
    }

    //Try to upgrade from no Motion degeneracy to t only or from R degeneracy to R+t
    if (enableUpgradeDegenPose && (degeneracyType & DEGEN_UPGRADE))
    {
        if (degeneracyType & DEGEN_NO_MOT)
        {
            //Upgrade no motion to t
            num_outliers = usac_num_data_points_ - usac_results_.degen_inlier_count_noMot;
            if (num_outliers < 1) {
                return 0;
            }

            std::vector<unsigned int> outlier_indices(num_outliers);
            unsigned int count = 0;
            for (unsigned int i = 0; i < usac_num_data_points_; ++i)
            {
                if (degen_outlier_flags_noMot[i])
                {
                    outlier_indices[count++] = i;
                }
            }
            std::vector<unsigned int> outlier_sample(1);
            //std::fill(errs_.begin(), errs_.begin() + usac_num_data_points_, DBL_MAX);
            std::fill(err_ptr_[0], err_ptr_[0] + usac_num_data_points_, DBL_MAX);
            auto current_err_array = err_ptr_[0];
            opengv::translation_t t_;
            unsigned int degen_sample_noMot_size = degen_sample_noMot.size();
            for (unsigned int i = 0; i < degen_max_upgrade_samples_noMot_trans; ++i)
            {
                generateUniformRandomSample(num_outliers, 1, &outlier_sample);
                std::vector<int> index(2);
                index[0] = (int)outlier_indices[outlier_sample[0]];
                unsigned int deg_sample_idx = std::rand() % degen_sample_noMot_size;
                index[1] = degen_sample_noMot[deg_sample_idx];

                t_ = opengv::relative_pose::twopt(*adapter_denorm, false, index);
                if (poselib::nearZero(t_.norm() * 100))
                    continue;

                unsigned int num_inliers, num_pts_tested;
                evaluateModelTrans(t_, &num_inliers, &num_pts_tested);

                if (num_inliers > best_upgrade_inliers_trans)
                {
                    for (unsigned int j = 0; j < 2; j++)
                        degen_sample_trans[j] = index[j];

                    // store translation
                    for (unsigned int r = 0; r < 3; ++r)
                        degen_final_model_params_trans[r] = t_(r);

                    t_eigen_upgrade = t_;

                    //If found translation has quiet high support compared to best inlier ratio of R+t
                    if (num_inliers > best_upgrade_inliers ||
                        (poselib::nearZero(final_model_params_[0] * 100) && poselib::nearZero(final_model_params_[4] * 100) && poselib::nearZero(final_model_params_[8] * 100)))
                    {
                        cv::Mat E_tmp, t_tmp, R_tmp = cv::Mat::eye(3, 3, CV_64FC1);
                        opengv::essential_t E_eigen;
                        cv::eigen2cv(t_, t_tmp);
                        E_tmp = poselib::getEfromRT(R_tmp, t_tmp);
                        cv::cv2eigen(E_tmp, E_eigen);
                        for (size_t r = 0; r < 3; r++)
                            for (size_t c = 0; c < 3; c++)
                                *(models_denorm_[0] + r * 3 + c) = E_eigen(r, c);
                        storeSolution(0, num_inliers);
                        best_upgrade_inliers = num_inliers;

                        if (degen_sample_noMot_size > 3)
                        {
                            unsigned int deg_sample_idx1 = 0;
                            for (size_t j = 0; j < 3; j++)
                            {
                                /*unsigned int deg_sample_idx1 = std::rand() % degen_sample_noMot_size;
                                while(degen_sample_noMot[deg_sample_idx1] == index[1])
                                    deg_sample_idx1 = std::rand() % degen_sample_noMot_size;
                                min_sample_[j] = degen_sample_noMot[deg_sample_idx1];*/

                                if (degen_sample_noMot[deg_sample_idx1] == index[1])
                                {
                                    j--;
                                    deg_sample_idx1++;
                                    continue;
                                }
                                min_sample_[j] = degen_sample_noMot[deg_sample_idx1];
                                deg_sample_idx1++;

                            }
                            min_sample_[3] = index[1];
                            min_sample_[4] = index[0];
                        }
                        usac_results_.degen_inlier_count_trans = num_inliers;
                    }

                    best_upgrade_inliers_trans = num_inliers;

                    unsigned int diffToNumOutliers = 0;
                    count = 0;
                    for (auto &j : outlier_indices)
                    {
                        if (*(current_err_array + j) < poseDegenTheshold)
                        {
                            ++count;
                        }
                        else if (std::round(*(current_err_array + j) - DBL_MAX) == 0)
                        {
                            ++diffToNumOutliers;
                        }
                    }
                    unsigned int num_samples = updateStandardStopping(count, num_outliers - diffToNumOutliers, 1);
                    if (num_samples < degen_max_upgrade_samples_noMot_trans)
                    {
                        degen_max_upgrade_samples_noMot_trans = num_samples;
                    }
                }
            }

            //Find a rotation
            //for (unsigned int i = 0; i < degen_max_upgrade_samples_noMot_rot; ++i)
            //{
            //	generateUniformRandomSample(num_outliers, 1, &outlier_sample);
            //	std::vector<int> index(2);
            //	index[0] = (int)outlier_indices[outlier_sample[0]];
            //	unsigned int deg_sample_idx = std::rand() % 5;
            //	index[1] = degen_sample_noMot[deg_sample_idx];

            //	R_ = opengv::relative_pose::twopt_rotationOnly(*adapter_denorm, index);

            //	unsigned int num_inliers, num_pts_tested;
            //	evaluateModelRot(R_, &num_inliers, &num_pts_tested);

            //	if (num_inliers > best_upgrade_inliers_rot)
            //	{
            //		for (unsigned int j = 0; j < 2; j++)
            //			degen_sample_rot[j] = (unsigned int)index[j];
            //
            //		// store rotation
            //		for (unsigned int r = 0; r < 3; ++r)
            //			for (unsigned int c = 0; c < 3; ++c)
            //				degen_final_model_params_rot[r * 3 + c] = R_(r, c);

            //		best_upgrade_inliers_rot = num_inliers;

            //		unsigned int count = 0, diffToNumOutliers = 0;;
            //		for (size_t j = 0; j < outlier_indices.size(); ++j)
            //		{
            //			if (*(current_err_array + outlier_indices[j]) < poseDegenTheshold)
            //			{
            //				++count;
            //			}
            //			else if (std::round(*(current_err_array + outlier_indices[j]) - DBL_MAX) == 0)
            //			{
            //				++diffToNumOutliers;
            //			}
            //		}
            //		unsigned int num_samples = updateStandardStopping(count, num_outliers - diffToNumOutliers, 1);
            //		if (num_samples < degen_max_upgrade_samples_noMot_rot)
            //		{
            //			degen_max_upgrade_samples_noMot_rot = num_samples;
            //		}

            //	}
            //
            //}
        }
        else
        {
            //Upgrade R to R+t
            num_outliers = usac_num_data_points_ - usac_results_.degen_inlier_count_rot;
            if (num_outliers < 3) {
                return 0;
            }

            std::vector<unsigned int> outlier_indices(num_outliers);
            unsigned int count = 0;
            for (unsigned int i = 0; i < usac_num_data_points_; ++i)
            {
                if (degen_outlier_flags_rot[i])
                {
                    outlier_indices[count++] = i;
                }
            }
            std::vector<unsigned int> outlier_sample(3);
            //std::fill(errs_.begin(), errs_.begin() + usac_num_data_points_, DBL_MAX);
            std::fill(err_ptr_[0], err_ptr_[0] + usac_num_data_points_, DBL_MAX);
            auto current_err_array = err_ptr_[0];
            opengv::rotation_t R_;
            opengv::translation_t t_;
            for (unsigned int i = 0; i < degen_max_upgrade_samples_rot; ++i)
            {
                generateUniformRandomSample(num_outliers, 3, &outlier_sample);
                std::vector<int> index(5);

                for (unsigned int j = 0; j < 3; j++)
                    index[j] = (int)outlier_indices[outlier_sample[j]];
                index[3] = degen_sample_rot[0];
                index[4] = degen_sample_rot[1];

                opengv::eigensolverOutput_t eig_out;
                cv::Mat R_tmp, t_tmp, E_tmp;
                adapter_denorm->setR12(R_eigen_degen);
                eig_out.rotation = R_eigen_degen;
                R_ = opengv::relative_pose::eigensolver(*adapter_denorm, index, eig_out);
                t_ = eig_out.translation;
                if (poselib::nearZero(t_.norm() * 100))
                    continue;
                t_ /= t_.norm();
                R_eigen_new = R_;
                t_eigen_new = t_;
                cv::eigen2cv(R_eigen_new, R_tmp);
                cv::eigen2cv(t_eigen_new, t_tmp);
                E_tmp = poselib::getEfromRT(R_tmp, t_tmp);
                for (size_t r = 0; r < 3; r++)
                    for (size_t c = 0; c < 3; c++)
                        *(models_denorm_[0] + r * 3 + c) = E_tmp.at<double>(r, c);

                unsigned int num_inliers, num_pts_tested;
                evaluateModel(0, &num_inliers, &num_pts_tested);

                if (num_inliers > best_upgrade_inliers_rot)
                {
                    count = 0;
                    for (unsigned int j = 2; j < 5; j++)
                        degen_sample_rot[j] = (unsigned int)index[count++];

                    //If found translation has quiet high support compared to best inlier ratio of R+t
                    if (num_inliers > best_upgrade_inliers ||
                        (poselib::nearZero(final_model_params_[0] * 100) && poselib::nearZero(final_model_params_[4] * 100) && poselib::nearZero(final_model_params_[8] * 100)))
                    {
                        storeSolution(0, num_inliers);
                        best_upgrade_inliers = num_inliers;
                        for (size_t j = 0; j < 5; j++)
                            min_sample_[j] = (unsigned int)degen_sample_rot[j];
                    }


                    best_upgrade_inliers_rot = num_inliers;

                    unsigned int diffToNumOutliers = 0;
                    count = 0;
                    for (auto &j : outlier_indices)
                    {
                        if (*(current_err_array + j) < usac_inlier_threshold_)
                        {
                            ++count;
                        }
                        else if (std::round(*(current_err_array + j) - DBL_MAX) == 0)
                        {
                            ++diffToNumOutliers;
                        }
                    }
                    unsigned int num_samples = updateStandardStopping(count, num_outliers - diffToNumOutliers, 1);
                    if (num_samples < degen_max_upgrade_samples_rot)
                    {
                        degen_max_upgrade_samples_rot = num_samples;
                    }
                }
            }
        }
    }

    std::cout << "Upgraded model has " << best_upgrade_inliers << " inliers" << std::endl;
    return best_upgrade_inliers;
}


// ============================================================================================
// findWeights: given model and points, compute weights to be used in local optimization
// ============================================================================================
void EssentialMatEstimator::findWeights(unsigned int modelIndex, const std::vector<unsigned int>& inliers,
    unsigned int numInliers, double* weights)
{
    double rx, ry, ryc, rxc;
    double* model;
    double* pt;
    unsigned int pt_index;

    if (refineMethod == USACConfig::REFINE_WEIGHTS)
    {
        model = models_[modelIndex];
        for (unsigned int i = 0; i < numInliers; ++i)
        {
            // get index of point to be verified
            pt_index = inliers[i];

            // compute weight (ref: torr dissertation, eqn. 2.25)
            pt = input_points_ + 6 * pt_index;
            rxc = (*model) * (*(pt + 3)) + (*(model + 3)) * (*(pt + 4)) + (*(model + 6));
            ryc = (*(model + 1)) * (*(pt + 3)) + (*(model + 4)) * (*(pt + 4)) + (*(model + 7));
            rx = (*model) * (*(pt)) + (*(model + 1)) * (*(pt + 1)) + (*(model + 2));
            ry = (*(model + 3)) * (*(pt)) + (*(model + 4)) * (*(pt + 1)) + (*(model + 5));

            weights[i] = 1 / sqrt(rxc*rxc + ryc*ryc + rx*rx + ry*ry);
        }
    }
    else if (refineMethod == USACConfig::REFINE_EIG_KNEIP_WEIGHTS)
    {
        model = models_denorm_[modelIndex];
        for (unsigned int i = 0; i < numInliers; ++i)
        {
            // get index of point to be verified
            pt_index = inliers[i];

            // compute weight (ref: torr dissertation, eqn. 2.25)
            pt = input_points_denorm_ + 6 * pt_index;
            rxc = (*model) * (*(pt + 3)) + (*(model + 3)) * (*(pt + 4)) + (*(model + 6));
            ryc = (*(model + 1)) * (*(pt + 3)) + (*(model + 4)) * (*(pt + 4)) + (*(model + 7));
            rx = (*model) * (*(pt)) + (*(model + 1)) * (*(pt + 1)) + (*(model + 2));
            ry = (*(model + 3)) * (*(pt)) + (*(model + 4)) * (*(pt + 1)) + (*(model + 5));

            weights[i] = 1 / sqrt(rxc*rxc + ryc*ryc + rx*rx + ry*ry);
        }
    }
    else if (refineMethod == USACConfig::REFINE_STEWENIUS_WEIGHTS || refineMethod == USACConfig::REFINE_NISTER_WEIGHTS)
    {
        opengv::essential_t modele = fivept_nister_essentials_denorm[modelIndex];
        opengv::bearingVector_t f, fprime;
        double pseudohuberth = std::sqrt(usac_inlier_threshold_) / 50.0;
        for (unsigned int i = 0; i < numInliers; ++i)
        {
            // get index of point to be verified
            //pt_index = inliers[i];

            // compute weight (pseudo-huber cost function)
            /*pt = input_points_denorm_ + 6 * pt_index;
            f << *pt, *(pt + 1), *(pt + 2);
            fprime << *(pt+3), *(pt + 4), *(pt + 5);*/
            f = adapter_denorm->getBearingVector2(inliers[i]);
            fprime = adapter_denorm->getBearingVector1(inliers[i]);
            weights[i] = computePseudoHuberWeight(f, fprime, modele, pseudohuberth);
        }
    }
}


// ============================================================================================
// storeModel: stores current best model
// this function is called  (by USAC) every time a new best model is found
// ============================================================================================
void EssentialMatEstimator::storeModel(unsigned int modelIndex, unsigned int numInliers)
{
    // save the current model as the best solution so far
    for (unsigned int i = 0; i < 9; ++i)
    {
        final_model_params_[i] = *(models_denorm_[modelIndex] + i);
    }
    R_eigen = R_eigen_new;
    t_eigen = t_eigen_new;	
}

#endif
