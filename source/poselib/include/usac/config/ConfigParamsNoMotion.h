#pragma once

#include "usac/config/ConfigParams.h"

namespace USACConfig
{
	//Possibilities to estimate the essential matrix:
	enum EssentialMatEstimator { ESTIM_NISTER, ESTIM_EIG_KNEIP, ESTIM_STEWENIUS };

	//Possibilities to refine the model:
	enum RefineAlgorithm {REFINE_WEIGHTS, REFINE_8PT_PSEUDOHUBER, REFINE_EIG_KNEIP, REFINE_EIG_KNEIP_WEIGHTS};

//Ampliotude for generating a random rotation (range: 0...1) that is used for initializing Kneip's Eigen solver
#define RAND_ROTATION_AMPLITUDE 0.1

	// problem specific/data-related parameters: fundamental matrix
	struct EssMat
	{
		EssMat() : refineMethod(REFINE_8PT_PSEUDOHUBER),
			hDegenThreshold(0.0),
			maxUpgradeSamples(500),
			used_estimator(ESTIM_NISTER)
		{}

		RefineAlgorithm		refineMethod;
		double				hDegenThreshold;
		unsigned int		maxUpgradeSamples;
		EssentialMatEstimator used_estimator;
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
		USACConfig::EssMat fund_,
		bool verbose_ = false) :
		ConfigParams(common_, prosac_, sprt_, losac_, verbose_),
		fund(fund_) {}

	USACConfig::EssMat fund;
};