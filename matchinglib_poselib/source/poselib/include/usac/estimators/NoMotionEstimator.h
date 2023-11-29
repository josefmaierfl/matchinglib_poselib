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
#include "usac/estimators/USAC.h"

#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeWeightingAdapter.hpp>
#include <opengv/math/cayley.hpp>
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

public:
	EssentialMatEstimator(std::mt19937 &mt) : USAC(mt), input_points_denorm_(nullptr), input_points_(nullptr), data_matrix_(nullptr), degen_data_matrix_(nullptr)
	{
		// input_points_ = NULL;
		// data_matrix_ = NULL;
		// degen_data_matrix_ = NULL;
		// models_.clear();
		// models_denorm_.clear();
	}

	~EssentialMatEstimator()
	{
		if (input_points_) { 
			delete[] input_points_;
			input_points_ = nullptr;
		}
		if (data_matrix_) { 
			delete[] data_matrix_;
			data_matrix_ = nullptr;
		}
		if (degen_data_matrix_) { 
			delete[] degen_data_matrix_;
			degen_data_matrix_ = nullptr;
		}
		for (size_t i = 0; i < models_.size(); ++i)
		{
			if (models_[i]) { delete[] models_[i]; }
		}
		models_.clear();
		for (size_t i = 0; i < models_denorm_.size(); ++i)
		{
			if (models_denorm_[i]) { delete[] models_denorm_[i]; }
		}
		models_denorm_.clear();
		adapter.reset();
	}

public:
	// ------------------------------------------------------------------------
	// problem specific functions
	inline void		 cleanupProblem();
	inline unsigned int generateMinimalSampleModels();
	inline bool		 generateRefinedModel(std::vector<unsigned int>& sample, const unsigned int numPoints,
		bool weighted = false, double* weights = NULL);
	inline bool		 validateSample();
	inline bool		 validateModel(unsigned int modelIndex);
	inline bool		 evaluateModel(unsigned int modelIndex, unsigned int* numInliers, unsigned int* numPointsTested);
	inline void		 testSolutionDegeneracy(bool* degenerateModel, bool* upgradeModel);
	inline unsigned int upgradeDegenerateModel();
	inline void		 findWeights(unsigned int modelIndex, const std::vector<unsigned int>& inliers,
		unsigned int numInliers, double* weights);
	inline void		 storeModel(unsigned int modelIndex, unsigned int numInliers);

private:
	double*		 input_points_denorm_;					    // stores pointer to original input points
	double       degen_homog_threshold_;				    // threshold for h-degen test
	unsigned int degen_max_upgrade_samples_;				// maximum number of upgrade attempts
	//USACConfig::MatrixDecomposition matrix_decomposition_method;  // QR/LU decomposition

																  // ------------------------------------------------------------------------
																  // temporary storage
	double* input_points_;								// stores normalized data points
	double* data_matrix_;								// linearized input data
	double* degen_data_matrix_;							// only for degeneracy testing
	std::vector<int> degen_outlier_flags_;				// for easy access to outliers to degeneracy
	double  m_T1_[9], m_T2_[9], m_T2_trans_[9];			// normalization matrices
	Eigen::Matrix3d m_T1_e, m_T2_e, m_T2_trans_e;		// normalization matrices in Eigen format
	Eigen::Matrix3d m_T1_e_inv, m_T2_e_inv, m_T2_trans_e_inv;		// inverse normalization matrices in Eigen format
	opengv::essentials_t fivept_nister_essentials;		// stores vector of models
	opengv::essentials_t fivept_nister_essentials_denorm; // stores vector of (denormalized) models
	opengv::rotation_t R_eigen;							// Stores a rotation estimated using Kneip's Eigen solver
	opengv::translation_t t_eigen;						// Stores a translation estimated using Kneip's Eigen solver
	std::vector<double*> models_;						// stores vector of models
	std::vector<double*> models_denorm_;				// stores vector of (denormalized) models

	std::unique_ptr<opengv::relative_pose::CentralRelativeAdapter> adapter; //Pointer to adapter for OpenGV
	std::unique_ptr<opengv::relative_pose::CentralRelativeAdapter> adapter_denorm; //Pointer to adapter for OpenGV if Kneip's Eigen solver is used
	USACConfig::RefineAlgorithm refineMethod; //Used method within generateRefinedModel
	USACConfig::EssentialMatEstimator used_estimator;//Specifies, which estimator from OpenGV should be used (Nister, Stewenius, Kneip's Eigen)
	cv::Mat p1, p2;
};

void getPerturbedRotation(
	opengv::rotation_t &rotation,
	std::mt19937 &mt,
	const double &amplitude)
{
	opengv::cayley_t cayley = opengv::math::rot2cayley(rotation);
	for (size_t i = 0; i < 3; i++)
	{
		cayley[i] =
			cayley[i] + (static_cast<double>(mt()) / static_cast<double>(mt.max()) - 0.5)*2.0*amplitude;
	}
	rotation = opengv::math::cayley2rot(cayley);
}


// ============================================================================================
// initProblem: initializes problem specific data and parameters
// this function is called once per run on new data
// ============================================================================================
bool EssentialMatEstimator::initProblem(const ConfigParamsEssential& cfg, double* pointData)
{
	// copy pointer to input data
	input_points_denorm_ = pointData;
	input_points_ = new double[6 * cfg.common.numDataPoints];
	if (input_points_denorm_ == NULL)
	{
		std::cerr << "Input point data not properly initialized" << std::endl;
		return false;
	}
	if (input_points_ == NULL)
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
	final_model_params_.clear(); final_model_params_.resize(9);
	models_.clear(); models_.resize(usac_max_solns_per_sample_);
	models_denorm_.clear(); models_denorm_.resize(usac_max_solns_per_sample_);
	for (unsigned int i = 0; i < usac_max_solns_per_sample_; ++i)
	{
		models_[i] = new double[9];
		models_denorm_[i] = new double[9];
	}

	opengv::bearingVectors_t bearingVectors1;
	opengv::bearingVectors_t bearingVectors2;
	opengv::bearingVectors_t bearingVectors1_denorm;
	opengv::bearingVectors_t bearingVectors2_denorm;
	p1 = cv::Mat(cfg.common.numDataPoints, 2, CV_64FC1);
	p2 = cv::Mat(cfg.common.numDataPoints, 2, CV_64FC1);
	double* p_idx1 = input_points_denorm_;
	double* p_idx2 = input_points_;
	for (unsigned int i = 0; i < cfg.common.numDataPoints; i++)
	{
		opengv::point_t bodyPoint1;// (input_points_ + i * 6); //--------------------->Check if this works
		opengv::point_t bodyPoint2;// (input_points_ + i * 6 + 3);
		bodyPoint1 << *(p_idx2++), *(p_idx2++), *(p_idx2++);//--------------------->Check if this works
		bodyPoint2 << *(p_idx2++), *(p_idx2++), *(p_idx2++);
		bodyPoint1 = bodyPoint1 / bodyPoint1.norm();
		bodyPoint2 = bodyPoint2 / bodyPoint2.norm();
		bearingVectors1.push_back(bodyPoint1);
		bearingVectors2.push_back(bodyPoint2);

		opengv::point_t bodyPoint1_d;// (input_points_denorm_ + i * 6); //--------------------->Check if this works
		opengv::point_t bodyPoint2_d;// (input_points_denorm_ + i * 6 + 3);
		bodyPoint1_d << *(p_idx1++), *(p_idx1++), *(p_idx1++);//--------------------->Check if this works
		bodyPoint2_d << *(p_idx1++), *(p_idx1++), *(p_idx1++);
		bodyPoint1_d = bodyPoint1_d / bodyPoint1_d.norm();
		bodyPoint2_d = bodyPoint2_d / bodyPoint2_d.norm();
		bearingVectors1_denorm.push_back(bodyPoint1_d);
		bearingVectors2_denorm.push_back(bodyPoint2_d);

		p1.at<double>(i, 0) = *(input_points_ + i * 6);
		p1.at<double>(i, 1) = *(input_points_ + i * 6 + 1);
		p2.at<double>(i, 0) = *(input_points_ + i * 6 + 3);
		p2.at<double>(i, 1) = *(input_points_ + i * 6 + 4);
	}

	//create a central relative adapter
	//opengv inverts the input inside the function (as Nister and Stewenius deliver a inverted pose), so invert the input here as well!!!!!!!!!!!
	adapter.reset(new opengv::relative_pose::CentralRelativeAdapter(
		bearingVectors2,
		bearingVectors1));

	adapter_denorm.reset(new opengv::relative_pose::CentralRelativeAdapter(
		bearingVectors2_denorm,
		bearingVectors1_denorm));

	fivept_nister_essentials = opengv::essentials_t(usac_max_solns_per_sample_);
	fivept_nister_essentials_denorm = opengv::essentials_t(usac_max_solns_per_sample_);


	// precompute the data matrix
	data_matrix_ = new double[9 * usac_num_data_points_];	// 9 values per correspondence
	FTools::computeDataMatrix(data_matrix_, usac_num_data_points_, input_points_);

	// if required, set up H-data matrix/storage for degeneracy testing
	degen_outlier_flags_.clear(); degen_outlier_flags_.resize(usac_num_data_points_, 0);
	if (usac_test_degeneracy_)//---------------------------------------------------> update below
	{
		degen_homog_threshold_ = cfg.fund.hDegenThreshold;
		degen_max_upgrade_samples_ = cfg.fund.maxUpgradeSamples;
		degen_final_model_params_.clear(); degen_final_model_params_.resize(9);
		degen_data_matrix_ = new double[2 * 9 * usac_num_data_points_];	// 2 equations per correspondence
		HTools::computeDataMatrix(degen_data_matrix_, usac_num_data_points_, input_points_);
	}
	else
	{
		degen_homog_threshold_ = 0.0;
		degen_max_upgrade_samples_ = 0;
		degen_final_model_params_.clear();
		degen_data_matrix_ = NULL;
	}

	// read in the f-matrix specific parameters from the config struct
	//matrix_decomposition_method = cfg.fund.decompositionAlg;
	refineMethod = cfg.fund.refineMethod;
	used_estimator = cfg.fund.used_estimator;
	degen_homog_threshold_ = cfg.fund.hDegenThreshold*cfg.fund.hDegenThreshold;

	return true;
}


// ============================================================================================
// cleanupProblem: release any temporary problem specific data storage
// this function is called at the end of each run on new data
// ============================================================================================
void EssentialMatEstimator::cleanupProblem()
{
	if (input_points_) { delete[] input_points_; input_points_ = NULL; }
	if (data_matrix_) { delete[] data_matrix_; data_matrix_ = NULL; }
	if (degen_data_matrix_) { delete[] degen_data_matrix_; degen_data_matrix_ = NULL; }
	for (size_t i = 0; i < models_.size(); ++i)
	{
		if (models_[i]) { delete[] models_[i]; }
	}
	models_.clear();
	for (size_t i = 0; i < models_denorm_.size(); ++i)
	{
		if (models_denorm_[i]) { delete[] models_denorm_[i]; }
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
		fivept_nister_essentials_denorm = opengv::relative_pose::fivept_nister(*adapter_denorm, indices);
		nsols = fivept_nister_essentials_denorm.size();
	}
	else if (used_estimator == USACConfig::ESTIM_EIG_KNEIP)
	{
		opengv::eigensolverOutput_t eig_out;
		opengv::essential_t E_eigen;
		cv::Mat R_tmp, t_tmp, E_tmp;
		//Variation of R as init for eigen-solver
		opengv::rotation_t R_init = Eigen::Matrix3d::Identity();
		getPerturbedRotation(R_init, mt_, RAND_ROTATION_AMPLITUDE); //Check if the amplitude is too large or too small!

		adapter_denorm->setR12(R_init);
		R_eigen = opengv::relative_pose::eigensolver(*adapter_denorm);
		t_eigen = eig_out.translation;
		cv::eigen2cv(R_eigen, R_tmp);
		cv::eigen2cv(t_eigen, t_tmp);
		E_tmp = poselib::getEfromRT(R_tmp, t_tmp);
		cv::cv2eigen(E_tmp, E_eigen);
		fivept_nister_essentials_denorm.clear();
		fivept_nister_essentials_denorm.push_back(E_eigen);
		nsols = 1;
	}
	else if (used_estimator == USACConfig::ESTIM_STEWENIUS)
	{
		opengv::complexEssentials_t fivept_stewenius_essentials;
		fivept_stewenius_essentials = opengv::relative_pose::fivept_stewenius(*adapter_denorm, indices);
		fivept_nister_essentials_denorm.clear();
		//Skip essential matrices with imaginary entries
		for (size_t i = 0; i < fivept_stewenius_essentials.size(); i++)
		{
			bool is_Imag = false;
			for (int r = 0; r < 3; r++)
			{
				for (int c = 0; c < 3; c++)
				{
					if (fivept_stewenius_essentials.at(i)(r, c).imag() != 0)
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
						E_eigen(r, c) = fivept_stewenius_essentials.at(i)(r, c).real();
				fivept_nister_essentials_denorm.push_back(E_eigen);
			}
		}

		nsols = fivept_nister_essentials_denorm.size();
	}
	else
	{
		std::cout << "Estimator not supported! Using Nister." << std::endl;
		fivept_nister_essentials_denorm = opengv::relative_pose::fivept_nister(*adapter_denorm, indices);
		nsols = fivept_nister_essentials_denorm.size();
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

	//double A[9 * 9];
	//
	//// form the matrix of equations for this minimal sample
	//double *src_ptr;
	//double *dst_ptr = A;
	//for (unsigned int i = 0; i < usac_min_sample_size_; ++i)
	//{
	//	src_ptr = data_matrix_ + min_sample_[i];
	//	for (unsigned int j = 0; j < 9; ++j)
	//	{
	//		*dst_ptr = *src_ptr;
	//		++dst_ptr;
	//		src_ptr += usac_num_data_points_;
	//	}
	//}

	//// LU/QR factorization
	//double sol[9 * 9];
	//double poly[4], roots[3];
	//double *f1, *f2;
	//int nullbuff[18];
	//f1 = sol;
	//f2 = sol + 9;
	//if (matrix_decomposition_method == USACConfig::DECOMP_QR)
	//{
	//	FTools::nullspaceQR7x9(A, sol);
	//}
	//else if (matrix_decomposition_method == USACConfig::DECOMP_LU)
	//{
	//	for (unsigned int i = 7 * 9; i < 9 * 9; ++i)
	//	{
	//		A[i] = 0.0;
	//	}
	//	int nullsize = FTools::nullspace(A, f1, 9, nullbuff);
	//	if (nullsize != 2)
	//	{
	//		return 0;
	//	}
	//}

	//// solve polynomial
	//FTools::makePolynomial(f1, f2, poly);
	//nsols = FTools::rroots3(poly, roots);

	//// form up to three fundamental matrices
	//double T2_F[9];
	//for (unsigned int i = 0; i < nsols; ++i)
	//{
	//	for (unsigned int j = 0; j < 9; ++j)
	//	{
	//		*(models_[i] + j) = f1[j] * roots[i] + f2[j] * (1 - roots[i]);
	//	}
	//	// store denormalized version as well
	//	MathTools::mmul(T2_F, m_T2_trans_, models_[i], 3);
	//	MathTools::mmul(models_denorm_[i], T2_F, m_T1_, 3);
	//}

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
	if (posetype == USACConfig::TRANS_ESSENTIAL)
	{
		if (refineMethod == USACConfig::REFINE_WEIGHTS)
		{
			// form the matrix of equations for this non-minimal sample
			double *A = new double[numPoints * 9];
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

			fivept_nister_essentials[0] = opengv::relative_pose::eightpt(*adapter, indices);
			fivept_nister_essentials_denorm[0] = m_T2_trans_e * fivept_nister_essentials[0] * m_T1_e;
			if (weighted)
			{
				cv::Mat E(3, 3, CV_64FC1);
				cv::eigen2cv(fivept_nister_essentials_denorm[0], E);
				cv::Mat mask((int)usac_num_data_points_, 1, CV_8U, false);
				for (unsigned int i = 0; i < numPoints; ++i)
				{
					mask.at<bool>((int)sample[i]) = true;
				}
				robustEssentialRefine(p1, p2, E, E, 0.02, 0, true, NULL, NULL, cv::noArray(), mask, 0, false, false);
				cv::cv2eigen(E, fivept_nister_essentials_denorm[0]);
			}
			fivept_nister_essentials[0] = m_T2_trans_e_inv * fivept_nister_essentials[0] * m_T1_e_inv;
			//Store result back into USAC format to allow using their validateSample() code
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
			if (used_estimator == USACConfig::ESTIM_EIG_KNEIP)
				adapter_denorm->setR12(R_eigen);
			else
			{
				R_init = Eigen::Matrix3d::Identity();
				getPerturbedRotation(R_init, mt_, RAND_ROTATION_AMPLITUDE); //Check if the amplitude is too large or too small!
				adapter_denorm->setR12(R_init);
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

				if (used_estimator == USACConfig::ESTIM_EIG_KNEIP)
					adapter_denorm_weights->setR12(R_eigen);
				else
					adapter_denorm_weights->setR12(R_init);

				R_eigen = opengv::relative_pose::eigensolver(*adapter_denorm_weights, eig_out);
				t_eigen = eig_out.translation;
			}
			else
			{
				std::vector<int> indices;
				for (unsigned int i = 0; i < numPoints; ++i)
				{
					indices.push_back((int)sample[i]);
				}
				R_eigen = opengv::relative_pose::eigensolver(*adapter_denorm, indices, eig_out);
				t_eigen = eig_out.translation;
			}

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
		else
		{
			std::cout << "Refinement algorithm not supported! Skipping!" << std::endl;
		}
	}
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
		getAnglesRotMat(R_tmp, roll, pitch, yaw, false);

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
	}

	return true;
}


// ============================================================================================
// validateSample: check if minimal sample is valid
// here, just returns true
// ============================================================================================
bool EssentialMatEstimator::validateSample()
{
	int j, k, i;

	for (i = 0; i < usac_min_sample_size_; i++)
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

	return i >= usac_min_sample_size_ - 1;

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
	std::vector<double>::iterator current_err_array = err_ptr_[0];
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
// test if >=5 points in the sample are on a plane
// ============================================================================================
void EssentialMatEstimator::testSolutionDegeneracy(bool* degenerateModel, bool* upgradeModel)
{
	*degenerateModel = false;
	*upgradeModel = false;

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
// upgradeDegenerateModel: try to upgrade degenerate model to non-degenerate by sampling from
// the set of outliers to the degenerate model
// ============================================================================================
unsigned int EssentialMatEstimator::upgradeDegenerateModel()
{
	unsigned int best_upgrade_inliers = usac_results_.best_inlier_count_;
	unsigned int num_outliers = usac_num_data_points_ - usac_results_.degen_inlier_count_;

	if (num_outliers < 2) {
		return 0;
	}

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
	std::vector<double>::iterator current_err_array = err_ptr_[0];

	double* pt1_index, *pt2_index;
	double x1[3], x1p[3], x2[3], x2p[3];
	double temp[3], l1[3], l2[3], ep[3];
	double skew_sym_ep[9];
	double T2_F[9];
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

		unsigned int num_inliers, num_pts_tested;
		evaluateModel(0, &num_inliers, &num_pts_tested);

		if (num_inliers > best_upgrade_inliers)
		{
			usac_results_.degen_sample_[3] = outlier_indices[outlier_sample[0]];//Overwrite samples on the plane with samples off the plane (for F (7pt alg), this samples are added)
			usac_results_.degen_sample_[4] = outlier_indices[outlier_sample[1]];//Overwrite samples on the plane with samples off the plane
			min_sample_ = usac_results_.degen_sample_;
			storeSolution(0, num_inliers);
			best_upgrade_inliers = num_inliers;

			unsigned int count = 0;
			for (size_t j = 0; j < outlier_indices.size(); ++j)
			{
				if (*(current_err_array + outlier_indices[j]) < usac_inlier_threshold_)
				{
					++count;
				}
			}
			unsigned int num_samples = updateStandardStopping(count, num_outliers, 2);
			//std::cout << "Inliers = " << num_inliers << ", in/out = " << count << "/" << num_outliers
			//	      << ". Num samples = " << num_samples << std::endl;
			if (num_samples < degen_max_upgrade_samples_)
			{
				degen_max_upgrade_samples_ = num_samples;
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
}

#endif
