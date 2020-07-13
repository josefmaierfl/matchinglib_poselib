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