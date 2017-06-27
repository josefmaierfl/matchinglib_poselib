#ifndef ROTATIONMATESTIMATOR_H
#define ROTATIONMATESTIMATOR_H

#include <iostream>
#include <fstream>
#include <string>
#include "usac/config/ConfigParamsRotationMat.h"
#include "usac/estimators/USAC.h"

#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <memory>


class RotationMatEstimator : public USAC<RotationMatEstimator>
{
public:
	inline bool		 initProblem(const ConfigParamsRotationMat& cfg, double* pointData);
	// ------------------------------------------------------------------------
	// storage for the final result
	opengv::rotation_t final_model_params_;
	opengv::rotation_t R_eigen;							// Stores a rotation

public:
	RotationMatEstimator()
	{
	};
	~RotationMatEstimator()
	{
		adapter_denorm.reset();
	};

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
	std::shared_ptr<opengv::relative_pose::CentralRelativeAdapter> adapter_denorm; //Pointer to adapter for OpenGV if Kneip's Eigen solver is used
	opengv::bearingVectors_t bearingVectors1_denorm;
	opengv::bearingVectors_t bearingVectors2_denorm;
};

// ============================================================================================
// initProblem: initializes problem specific data and parameters
// this function is called once per run on new data
// ============================================================================================
bool RotationMatEstimator::initProblem(const ConfigParamsRotationMat& cfg, double* pointData)
{
	// copy pointer to input data
	input_points_denorm_ = pointData;
	if (input_points_denorm_ == NULL)
	{
		std::cerr << "Input point data not properly initialized" << std::endl;
		return false;
	}
	double* p_idx1 = input_points_denorm_;
	for (unsigned int i = 0; i < cfg.common.numDataPoints; i++)
	{
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

	adapter_denorm.reset(new opengv::relative_pose::CentralRelativeAdapter(
		bearingVectors2_denorm,
		bearingVectors1_denorm));

	return true;
}


// ============================================================================================
// cleanupProblem: release any temporary problem specific data storage 
// this function is called at the end of each run on new data
// ============================================================================================
void RotationMatEstimator::cleanupProblem()
{
	
}


// ============================================================================================
// generateMinimalSampleModels: generates minimum sample model(s) from the data points whose  
// indices are currently stored in m_sample. 
// the generated models are stored in a vector of models and are all evaluated
// ============================================================================================
unsigned int RotationMatEstimator::generateMinimalSampleModels()
{
	unsigned int nsols = 0;
	std::vector<int> indices;
	for (unsigned int i = 0; i < usac_min_sample_size_; ++i)
	{
		indices.push_back((int)min_sample_[i]);
	}

	R_eigen = opengv::relative_pose::twopt_rotationOnly(*adapter_denorm, indices);
	nsols = 1;

	return nsols;
}


// ============================================================================================
// generateRefinedModel: compute model using non-minimal set of samples
// default operation is to use a weight of 1 for every data point
// ============================================================================================
bool RotationMatEstimator::generateRefinedModel(std::vector<unsigned int>& sample,
	const unsigned int numPoints,
	bool weighted,
	double* weights)
{
	std::vector<int> indices;
	for (unsigned int i = 0; i < numPoints; ++i)
	{
		indices.push_back((int)sample[i]);
	}
	R_eigen = opengv::relative_pose::rotationOnly(*adapter_denorm, indices);
	
	return true;
}


// ============================================================================================
// validateSample: check if minimal sample is valid
// here, just returns true
// ============================================================================================
bool RotationMatEstimator::validateSample()
{
	return true;
}


// ============================================================================================
// validateModel: check if model computed from minimal sample is valid
// checks oriented constraints to determine model validity
// ============================================================================================
bool RotationMatEstimator::validateModel(unsigned int modelIndex)//---------------------------------------> does this work with E?
{
	return true;
}


// ============================================================================================
// evaluateModel: test model against all/subset of the data points
// ============================================================================================
bool RotationMatEstimator::evaluateModel(unsigned int modelIndex, unsigned int* numInliers, unsigned int* numPointsTested)
{
	double temp_err;
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

		opengv::bearingVector_t f1 = adapter_denorm->getBearingVector1(pt_index);
		opengv::bearingVector_t f2 = adapter_denorm->getBearingVector2(pt_index);

		//unrotate bearing-vector f2
		opengv::bearingVector_t f2_unrotated = R_eigen * f2;

		//bearing-vector based outlier criterium (select threshold accordingly):
		//1-(f1'*f2) = 1-cos(alpha) \in [0:2]
		temp_err = 1.0 - (f1.transpose() * f2_unrotated);
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
// test if >=5 points in the sample are on a plane, or if the pose is only a rotation, translation, or no motion
// ============================================================================================
void RotationMatEstimator::testSolutionDegeneracy(bool* degenerateModel, bool* upgradeModel)
{
	*degenerateModel = false;
	*upgradeModel = false;
}


// ============================================================================================
// upgradeDegenerateModel: try to upgrade degenerate model to non-degenerate by sampling from
// the set of outliers to the degenerate model
// ============================================================================================
unsigned int RotationMatEstimator::upgradeDegenerateModel()
{
	return 0;
}


// ============================================================================================
// findWeights: given model and points, compute weights to be used in local optimization
// ============================================================================================
void RotationMatEstimator::findWeights(unsigned int modelIndex, const std::vector<unsigned int>& inliers,
	unsigned int numInliers, double* weights)
{
	for (unsigned int i = 0; i < numInliers; ++i)
	{
		weights[i] = 1.0;
	}
}


// ============================================================================================
// storeModel: stores current best model
// this function is called  (by USAC) every time a new best model is found
// ============================================================================================
void RotationMatEstimator::storeModel(unsigned int modelIndex, unsigned int numInliers)
{
	// save the current model as the best solution so far
	final_model_params_ = R_eigen;
}

#endif

