#ifndef CONFIGPARAMSFUND_H
#define CONFIGPARAMSFUND_H

#include "usac/config/ConfigParams.h"

namespace USACConfig
{
	enum MatrixDecomposition      {DECOMP_QR, DECOMP_LU};

	// problem specific/data-related parameters: fundamental matrix
	struct Fund
	{
		Fund(): decompositionAlg  (DECOMP_QR),
			hDegenThreshold       (0.0),
			maxUpgradeSamples     (500)//,
			//inputFilePath	      ("")			// leave blank if not using config file
		{}

		MatrixDecomposition decompositionAlg;
		double				hDegenThreshold;
		unsigned int		maxUpgradeSamples;
		//std::string			inputFilePath;
	};
}

class ConfigParamsFund: public ConfigParams
{
public:
	// simple function to read in parameters from config file
	//bool initParamsFromConfigFile(std::string& configFilePath);
	
	ConfigParamsFund(USACConfig::Common common_,
					USACConfig::Prosac prosac_,
					USACConfig::Sprt sprt_,
					USACConfig::Losac losac_,
					USACConfig::Fund fund_,
					bool verbose_ = false):
					ConfigParams(common_, prosac_, sprt_, losac_, verbose_),
					fund(fund_)	{
		if ((fund.decompositionAlg != USACConfig::DECOMP_LU) &&
			(fund.decompositionAlg != USACConfig::DECOMP_QR))
		{
			std::cerr << "Matrix decomposition " << fund.decompositionAlg << " not supported. Setting it to USACConfig::DECOMP_LU" << std::endl;
			fund.decompositionAlg = USACConfig::DECOMP_LU;
		}
	}

	USACConfig::Fund fund;
};

#endif