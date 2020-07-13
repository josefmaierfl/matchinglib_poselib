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