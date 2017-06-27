#pragma once

#include "usac/config/ConfigParams.h"

namespace USACConfig
{
	//Possibilities to estimate the essential matrix:
	enum EssentialMatEstimatorUsed { ESTIM_NISTER, ESTIM_EIG_KNEIP, ESTIM_STEWENIUS };

	//Possibilities to refine the model:
	enum RefineAlgorithm {
		REFINE_WEIGHTS, 
		REFINE_8PT_PSEUDOHUBER, 
		REFINE_EIG_KNEIP, 
		REFINE_EIG_KNEIP_WEIGHTS, 
		REFINE_STEWENIUS, 
		REFINE_STEWENIUS_WEIGHTS,
		REFINE_NISTER,
		REFINE_NISTER_WEIGHTS
	};

//Ampliotude for generating a random rotation (range: 0...1) that is used for initializing Kneip's Eigen solver
#define RAND_ROTATION_AMPLITUDE 0.1
	//Maximum solutions (trials) from Kneip's Eigen solver
#define MAX_SOLS_KNEIP 12

	// problem specific/data-related parameters: essential matrix
	struct EssMat
	{
		EssMat() : refineMethod(REFINE_8PT_PSEUDOHUBER),
			hDegenThreshold(0.0),
			maxUpgradeSamples(500),
			used_estimator(ESTIM_NISTER),
			focalLength(800),
			rotDegenThesholdPix(0.8),
			ransacLikeUpgradeDegenerate(true),
			enableHDegen(false),
			enableUpgradeDegenPose(true),
			completeModel(false)
		{}

		RefineAlgorithm		refineMethod;
		double				hDegenThreshold;
		unsigned int		maxUpgradeSamples;
		EssentialMatEstimatorUsed used_estimator;
		double				focalLength;
		double				rotDegenThesholdPix;
		bool				ransacLikeUpgradeDegenerate;
		bool				enableHDegen;
		bool				enableUpgradeDegenPose;
		bool				completeModel;		
	};
}

class ConfigParamsEssential : public ConfigParams
{
public:
	// simple function to read in parameters from config file
	//bool initParamsFromConfigFile(std::string& configFilePath);

	ConfigParamsEssential(USACConfig::Common common_,
		USACConfig::Prosac prosac_,
		USACConfig::Sprt sprt_,
		USACConfig::Losac losac_,
		USACConfig::EssMat fund_) :
		ConfigParams(common_, prosac_, sprt_, losac_),
		fund(fund_) {}

	USACConfig::EssMat fund;
};