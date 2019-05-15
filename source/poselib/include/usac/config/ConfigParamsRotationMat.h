#pragma once

#include "usac/config/ConfigParams.h"

//namespace USACConfig
//{
//	// problem specific/data-related parameters: fundamental matrix
//	struct RotMat
//	{
//		RotMat() : focalLength(800),
//			rotDegenThesholdPix(0.8)
//		{}
//
//		double				focalLength;
//		double				rotDegenThesholdPix;		
//	};
//}

class ConfigParamsRotationMat : public ConfigParams
{
public:
	// simple function to read in parameters from config file
	//bool initParamsFromConfigFile(std::string& configFilePath);

	ConfigParamsRotationMat(USACConfig::Common common_,
		USACConfig::Prosac prosac_,
		USACConfig::Sprt sprt_,
		USACConfig::Losac losac_,
		bool verbose_ = false) :
		ConfigParams(common_, prosac_, sprt_, losac_, verbose_)/*,
		fund(fund_)*/ {}

	//USACConfig::RotMat fund;
};